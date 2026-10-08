"""Explicit runtime population views shared by training and evaluation."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
from iblatlas.regions import BrainRegions
from utils.config_utils import load_config


def select_neural_regions(views, metadata, selection="all_recorded", *, regions=None):
    """Retain source order, identities and split membership while selecting units."""
    if selection not in ("all_recorded", "visual_only"):
        raise ValueError(f"Unsupported neural region selection: {selection}")
    if selection == "all_recorded":
        return (*views, metadata)
    metadata = deepcopy(metadata)
    metadata["neural_region_selection"] = selection
    regions = (load_config(Path(__file__).resolve().parents[1] / "configs/visual-regions.json")["regions"]
               if regions is None else regions)
    atlas = BrainRegions()
    if not regions or any(name not in atlas.acronym for name in regions):
        raise ValueError("Visual selection requires known Allen region acronyms")
    effective_ids = set()
    for name in regions:
        for atlas_id in np.unique(np.abs(atlas.acronym2id(name))):
            effective_ids.update(abs(int(value)) for value in atlas.descendants(int(atlas_id)).id)
    policy = dict(regions=list(regions), include_descendants=True, atlas="Allen CCF 2017",
                  effective_atlas_ids=sorted(effective_ids))
    metadata["neural_selection_policy"] = policy
    indices = {}
    tables = {}
    for view in views:
        for sample in view.samples:
            table = sample.neuron_identity
            if "atlas_id" not in table:
                raise ValueError("Visual-only selection requires Allen atlas IDs")
            if sample.session_id in tables:
                if not table.equals(tables[sample.session_id]):
                    raise ValueError("Ordered source populations differ within a session")
                continue
            selected = np.flatnonzero(table["atlas_id"].abs().isin(effective_ids).to_numpy())
            if not len(selected):
                raise ValueError(f"No configured visual-region neurons in session {sample.session_id}")
            indices[sample.session_id] = selected
            tables[sample.session_id] = table
    result = []
    for view in views:
        samples = []
        for sample in view.samples:
            selected = indices[sample.session_id]
            provenance = deepcopy(sample.metadata)
            provenance["runtime_neural_selection"] = dict(
                selection=selection, policy=deepcopy(policy),
                source_neuron_count=sample.neuron_count, source_columns=selected.tolist())
            samples.append(replace(sample, neural=sample.neural[:, selected].copy(),
                                   neuron_identity=sample.neuron_identity.iloc[selected].reset_index(drop=True).copy(),
                                   neuron_mask=sample.neuron_mask[selected].copy(),
                                   neuron_count=len(selected), metadata=provenance))
        result.append(replace(view, samples=tuple(samples)))
    metadata["eid_list"] = {session: len(indices[session]) for session in metadata["eid_list"]}
    metadata["num_neurons"] = sorted(set(metadata["eid_list"].values()))
    return (*result, metadata)
