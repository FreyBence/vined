"""Explicit runtime anatomical count targets; published samples stay unchanged."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
from iblatlas.regions import BrainRegions
from training_dataset import temporal_context_view


def _recorded_groups(sample, metadata, atlas):
    """Derive disjoint area groups only from the already selected population."""
    roots = metadata.get("neural_selection_policy", {}).get("regions", [])
    root_ids = {region: set(abs(int(i)) for i in atlas.descendants(
                abs(int(atlas.acronym2id(region)[0]))).id) for region in roots}
    labels = {}
    for column, atlas_id in enumerate(sample.neuron_identity["atlas_id"]):
        if not sample.neuron_mask[column]:
            continue
        if pd.isna(atlas_id):
            raise ValueError("Regional aggregation requires anatomical IDs for every valid recorded neuron")
        atlas_id = abs(int(atlas_id))
        matches = [region for region, ids in root_ids.items() if atlas_id in ids]
        if roots:
            if len(matches) != 1:
                raise ValueError("Selected visual neuron must belong to exactly one configured visual area")
            region = matches[0]
        else:
            indices = np.flatnonzero(atlas.id == atlas_id)
            if not len(indices):
                raise ValueError(f"Unknown recorded Allen atlas ID: {atlas_id}")
            index = int(indices[0])
            # Cortical layers are recorded subdivisions of the same target area.
            while "layer" in str(atlas.name[index]).lower():
                parents = np.flatnonzero(atlas.id == atlas.parent[index])
                if not len(parents):
                    raise ValueError("Recorded cortical layer has no anatomical parent area")
                index = int(parents[0])
            region = str(atlas.acronym[index])
        labels.setdefault(region, []).append(column)
    if not labels:
        raise ValueError(f"No valid recorded regional neurons in {sample.session_id}")
    names = sorted(labels)
    return names, [np.asarray(labels[name], dtype=np.int64) for name in names]


def transform_neural_targets(views, metadata, granularity="neuron", *, regions=None, selection=None):
    """Sum recorded units by parent area or recorded anatomical subdivision."""
    if selection is not None and (granularity != "subregion" or regions is not None):
        raise ValueError("Explicit source-column selection requires subregion encoding")
    if granularity == "neuron":
        return (*views, metadata)
    if granularity == "subregion" and regions is not None:
        raise ValueError("Subregion targets derive their parent partition from the selected population; explicit roots are unsupported")
    if granularity not in ("region", "subregion") or (regions is not None and (
            not regions or len(set(regions)) != len(regions))):
        raise ValueError("Region targets require unique explicit Allen root acronyms")
    atlas = BrainRegions()
    groups = []
    for region in regions or []:
        if region not in atlas.acronym:
            raise ValueError(f"Unknown Allen region: {region}")
        root = abs(int(atlas.acronym2id(region)[0]))
        groups.append(set(abs(int(i)) for i in atlas.descendants(root).id))
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError("Regional target groups must not overlap")
    metadata = deepcopy(metadata)
    populations, mappings, session_groups = {}, {}, {}
    result = []
    for view in views:
        samples = []
        for sample in view.samples:
            table = sample.neuron_identity
            if "atlas_id" not in table or sample.neural.dtype != np.int64:
                raise ValueError("Regional aggregation requires anatomical IDs and original int64 counts")
            if regions is None:
                if sample.session_id not in session_groups:
                    session_groups[sample.session_id] = _recorded_groups(sample, metadata, atlas)
                names, columns = session_groups[sample.session_id]
                if granularity == "subregion":
                    # Refine the existing parent partition using recorded Allen IDs,
                    # including unsplit noncortical areas and area-only labels.
                    refined = {}
                    for parent, indices in zip(names, columns):
                        for column in indices:
                            atlas_id = abs(int(table.iloc[column].atlas_id))
                            matches = np.flatnonzero(atlas.id == atlas_id)
                            if not len(matches):
                                raise ValueError(f"Unknown recorded Allen atlas ID: {atlas_id}")
                            label = str(atlas.acronym[int(matches[0])])
                            refined.setdefault((parent, label), []).append(int(column))
                    keys = sorted(refined)
                    parents = [key[0] for key in keys]
                    names = [key[1] for key in keys]
                    if len(set(names)) != len(names):
                        raise ValueError("Recorded anatomical label belongs to multiple parent regions")
                    columns = [np.asarray(refined[key], dtype=np.int64) for key in keys]
                expected_valid = sorted(int(c) for group in columns for c in group)
                if expected_valid != np.flatnonzero(sample.neuron_mask[:len(table)]).tolist():
                    raise ValueError("Recorded-neuron validity differs within a session")
            else:
                names = list(regions)
                columns = [np.flatnonzero(table.atlas_id.abs().isin(ids).to_numpy()
                           & sample.neuron_mask[:len(table)]) for ids in groups]
            if selection is not None:
                chosen = selection.get(sample.session_id)
                if not isinstance(chosen, dict) or not chosen or set(chosen) - set(names):
                    raise ValueError("Selection requires known, nonempty session subregion mappings")
                retained = []
                for label, parent, original in zip(names, parents, columns):
                    if label not in chosen:
                        continue
                    indices = chosen[label]
                    if (not isinstance(indices, list) or not indices or
                            any(type(c) is not int for c in indices) or
                            indices != sorted(set(indices)) or not set(indices) <= set(original.tolist())):
                        raise ValueError("Selected columns must be unique ordered members of their anatomical group")
                    retained.append((label, parent, np.asarray(indices, dtype=np.int64)))
                names, parents, columns = map(list, zip(*retained))
            if any(not len(indices) for indices in columns):
                raise ValueError(f"Empty recorded regional population in {sample.session_id}")
            mapping = dict(source_neuron_count=sample.neuron_count,
                           ordered_units=table.astype(object).where(pd.notna(table), None).to_dict(orient="records"),
                           regions=names, source_columns=[c.tolist() for c in columns],
                           region_neuron_counts=[len(c) for c in columns])
            if granularity == "subregion":
                mapping.update(parent_regions=parents, aggregation="sum", units="spike_count_per_bin",
                               temporal_indices="unchanged original sample.temporal_positions")
            if sample.session_id in populations and not table.equals(populations[sample.session_id]):
                raise ValueError("Source neuron ordering differs within a session")
            if sample.session_id in mappings and mapping["source_columns"] != mappings[sample.session_id]["source_columns"]:
                raise ValueError("Regional recorded-neuron mapping differs within a session")
            populations[sample.session_id], mappings[sample.session_id] = table, mapping
            neural = np.stack([sample.neural[:, c].sum(axis=1, dtype=np.int64)
                               for c in columns], axis=-1)
            neural[~sample.temporal_mask] = 0
            provenance = deepcopy(sample.metadata)
            provenance["regional_targets"] = mapping
            identity = pd.DataFrame(dict(acronym=names,
                uuids=[f"{sample.session_id}:{granularity}:{r}" for r in names],
                source_neuron_count=[len(c) for c in columns]))
            if granularity == "subregion":
                identity["parent_region"] = parents
            samples.append(replace(sample, neural=neural, neuron_identity=identity,
                neuron_count=len(names), neuron_mask=np.ones(len(names), dtype=bool), metadata=provenance))
        result.append(replace(view, samples=tuple(samples)))
    widths = {s: len(mappings[s]["regions"]) for s in metadata["eid_list"]}
    metadata.update(target_granularity=granularity, regional_targets=mappings,
                    eid_list=widths, num_neurons=sorted(set(widths.values())))
    if regions is None:
        metadata["regional_target_policy"] = dict(
            source="scenario_selected_recorded_population", ordering="alphabetical_per_session",
            anatomical_groups="configured visual roots" if metadata.get("neural_selection_policy")
                              else "recorded Allen areas with cortical layers merged into their parent area")
        if granularity == "subregion":
            metadata["regional_target_policy"]["anatomical_groups"] = "recorded Allen labels within the existing parent-area partition; no activity filtering"
    return (*result, metadata)


def regional_training_means(train):
    """Count-weighted constant rates using only real, eligible H=12 train bins."""
    sums, bins = {}, {}
    for sample in train.samples:
        if sample.split != "train":
            raise ValueError("Baseline statistics require training samples only")
        valid = temporal_context_view(sample.temporal_positions, sample.temporal_mask,
            direction="encoding", mode="full_trial").target_mask[0].numpy()
        session = sample.session_id
        sums[session] = sums.get(session, 0) + sample.neural[valid].sum(axis=0, dtype=np.int64)
        bins[session] = bins.get(session, 0) + int(valid.sum())
    if any(n == 0 for n in bins.values()):
        raise ValueError("No eligible training targets for regional bias")
    return {s: dict(mean_counts=(sums[s] / bins[s]).tolist(), eligible_bins=bins[s],
                    spike_sums=sums[s].tolist(), split="train") for s in sums}
