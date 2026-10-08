"""Validate and select existing ALF sorting results without recomputing metrics."""

from dataclasses import asdict, dataclass, field
from importlib.metadata import version

import numpy as np
import pandas as pd
from iblatlas.regions import BrainRegions
from .provenance import content_hash
from utils.progress import logger


@dataclass(frozen=True)
class Coverage:
    """Caller-supplied session-second intervals and their evidence/qualification."""

    observed: tuple | None = None
    invalid: tuple = ()
    provenance: str | None = None
    qualification: str = "unknown"


@dataclass(frozen=True)
class RecordingRequest:
    pid: str | None = None
    pname: str | None = None
    collection: str | None = None
    revision: str | None = None
    coverage: Coverage = field(default_factory=Coverage)


@dataclass(frozen=True)
class QualitySelection:
    """Filter a supplied field by source values or a minimum, with its meaning."""

    field: str
    interpretation: str
    values: tuple | None = None
    minimum: float | None = None


@dataclass(frozen=True)
class RegionSelection:
    acronyms: tuple
    mapping: str = "Allen"
    include_descendants: bool = False


@dataclass
class Population:
    eid: str
    spikes: dict
    units: pd.DataFrame
    recordings: tuple
    selection: dict

    @property
    def status(self):
        return "empty_selection" if self.units.empty else "available"


def _vector(value, name, length=None, integer=False):
    value = np.asarray(value)
    if value.ndim != 1 or (length is not None and len(value) != length):
        raise ValueError(f"{name} must be a vector with matching source rows")
    if integer and value.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer source indices")
    return value


def _coverage(value):
    if value.qualification not in ("unknown", "observed", "assumed"):
        raise ValueError("Coverage qualification must be unknown, observed, or assumed")
    if value.observed is not None and value.qualification == "unknown":
        raise ValueError("Supplied observed intervals need a coverage qualification")
    if value.observed is None and value.qualification != "unknown":
        raise ValueError("Qualified coverage requires explicit observed intervals")
    if (value.observed is not None or len(value.invalid)) and not value.provenance:
        raise ValueError("Coverage intervals require evidence or an explicit assumption")
    for name, intervals in (("observed", value.observed), ("invalid", value.invalid)):
        if intervals is None:
            continue
        array = np.asarray(intervals, dtype=float)
        if not array.size and array.shape == (0,):
            continue
        if (array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all()
                or np.any(array[:, 1] <= array[:, 0])):
            raise ValueError(f"Invalid {name} coverage intervals")
    return asdict(value)


def _unit_table(clusters, channels):
    channel_indices = _vector(clusters["channels"], "clusters.channels", integer=True)
    count = len(channel_indices)
    if np.any(channel_indices < 0):
        raise ValueError("Negative cluster channel reference")
    channel_lengths = set()
    for name, values in channels.items():
        values = np.asarray(values)
        if values.ndim == 0:
            raise ValueError(f"channels.{name} must have source channel rows")
        channel_lengths.add(len(values))
    if len(channel_lengths) != 1 or np.any(channel_indices >= next(iter(channel_lengths), 0)):
        raise ValueError("Inconsistent channel rows or unresolvable cluster channel reference")
    table = pd.DataFrame({"source_unit_id": np.arange(count, dtype=np.int64),
                          "channels": channel_indices.copy()})
    for name, values in clusters.items():
        if name in ("channels", "metrics"):
            continue
        array = np.asarray(values)
        if array.ndim == 0 or len(array) != count:
            raise ValueError(f"clusters.{name} does not match source unit rows")
        if name in ("source_unit_id", "eid", "pid", "probe", "collection", "revision", "recording_index", "selected_region"):
            raise ValueError(f"Reserved neural identity field in source clusters: {name}")
        table[name] = array.copy() if array.ndim == 1 else list(array.copy())

    metrics = clusters.get("metrics")
    if metrics is not None:
        metrics = pd.DataFrame(metrics).copy()
        if "cluster_id" in metrics:
            ids = _vector(metrics["cluster_id"], "metrics.cluster_id", integer=True)
            if len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= count)):
                raise ValueError("Ambiguous cluster metric identities")
            metrics = metrics.set_index("cluster_id").reindex(range(count))
        elif not metrics.index.equals(pd.RangeIndex(count)):
            raise ValueError("Metrics require source row order or explicit cluster_id")
        for name in metrics:
            if name in ("source_unit_id", "eid", "pid", "probe", "collection", "revision", "recording_index", "selected_region"):
                raise ValueError(f"Reserved neural identity field in metrics: {name}")
            if name in table:
                existing, supplied = table[name], metrics[name].reset_index(drop=True)
                same = existing.eq(supplied) | (existing.isna() & supplied.isna())
                if not same.all():
                    raise ValueError(f"Conflicting supplied cluster metric: {name}")
            else:
                table[name] = metrics[name].to_numpy(copy=True)

    # Enrich only from available channel fields; no metric recomputation or
    # fabricated anatomy. Direct cluster fields take precedence over channel data.
    for source_name, target in (("brainLocationIds_ccf_2017", "atlas_id"),
                                ("atlas_id", "atlas_id"), ("acronym", "acronym")):
        if target not in table and source_name in clusters:
            table[target] = table[source_name]
        if target not in table and source_name in channels:
            values = _vector(channels[source_name], f"channels.{source_name}")
            if np.any(channel_indices >= len(values)):
                raise ValueError(f"Unresolvable channel reference in {source_name}")
            table[target] = values[channel_indices]
    if "uuids" in table:
        known = table["uuids"].dropna()
        if known.duplicated().any():
            raise ValueError("Duplicate source unit UUIDs")
    for name in ("depths", "uuids", "label", "atlas_id", "acronym"):
        if name not in table:
            table[name] = None
    return table


def _anatomy(table, regions, policy):
    """Keep unknown anatomy unknown, and resolve only recognized atlas entries."""
    atlas_ids = pd.to_numeric(table["atlas_id"], errors="raise").to_numpy(dtype=float)
    supplied_ids = table["atlas_id"].notna().to_numpy()
    if np.any(supplied_ids & (~np.isfinite(atlas_ids) | (atlas_ids != np.floor(atlas_ids)))):
        raise ValueError("Supplied atlas IDs must be finite integers or unknown")
    known = np.isfinite(atlas_ids) & np.isin(atlas_ids, regions.id) & (atlas_ids != 0)
    if known.any():
        resolved_names = regions.id2acronym(atlas_ids[known].astype(np.int64), mapping="Allen")
        supplied_names = table.loc[known, "acronym"]
        if (supplied_names.notna() & supplied_names.ne(resolved_names)).any():
            raise ValueError("Supplied cluster acronym contradicts its Allen atlas ID")
        table.loc[known, "acronym"] = resolved_names
    if policy is None:
        return np.ones(len(table), dtype=bool), None
    if policy.mapping not in regions.mappings:
        raise ValueError(f"Unknown atlas mapping: {policy.mapping}")
    requested = tuple(policy.acronyms)
    if not requested or not all(isinstance(a, str) and a in regions.acronym and a != "void" for a in requested):
        raise ValueError("Anatomical selection requires known non-void region acronyms")
    mapped = np.full(len(table), None, dtype=object)
    if known.any():
        mapped[known] = regions.id2acronym(atlas_ids[known].astype(np.int64), mapping=policy.mapping)
    # A supplied Allen acronym can be used when an atlas ID is unavailable.
    names = table["acronym"].to_numpy(dtype=object)
    named = ~supplied_ids & pd.notna(names) & np.isin(names, regions.acronym) & (names != "void")
    if named.any():
        mapped[named] = regions.acronym2acronym(names[named], mapping=policy.mapping)
    if len(table) and not (known | named).any():
        raise ValueError("Anatomical selection requires supplied anatomical assignments")
    mapped_region_names = set(regions.acronym[regions.mappings[policy.mapping]])
    if not set(requested).issubset(mapped_region_names):
        raise ValueError("Requested region is not represented in the selected mapping")
    effective = set(requested)
    if policy.include_descendants:
        for name in requested:
            ids = regions.id[regions.acronym == name]
            descendants = regions.descendants(ids).id
            effective.update(regions.id2acronym(descendants, mapping=policy.mapping).tolist())
    table["selected_region"] = mapped
    return np.isin(mapped, sorted(effective)), dict(
        **asdict(policy), atlas="Allen CCF 2017", iblatlas_version=version("iblatlas"),
        effective_regions=sorted(effective))


def _quality_mask(table, policy):
    if policy is None:
        return np.ones(len(table), dtype=bool)
    if not policy.interpretation.strip() or (policy.values is None) == (policy.minimum is None):
        raise ValueError("Quality selection needs an interpretation and either values or minimum")
    if policy.field not in table:
        raise ValueError(f"Required quality field unavailable: {policy.field}")
    values = table[policy.field]
    if len(values) and values.isna().all():
        raise ValueError(f"Required quality field unavailable: {policy.field}")
    if policy.values is not None:
        return (values.notna() & values.isin(policy.values)).to_numpy()
    if not np.isfinite(policy.minimum):
        raise ValueError("Quality minimum must be finite")
    numeric = pd.to_numeric(values, errors="raise")
    return (numeric.notna() & np.isfinite(numeric) & (numeric >= policy.minimum)).to_numpy()


def load_population(access, eid, recordings, *, quality=None, anatomy=None):
    """Load explicitly requested recordings in canonical order; see interface.md.

    Spike assignments are positions in the returned unit table. source_unit_id
    remains the original ALF cluster row, scoped by recording and sorting source.
    Missing source errors from SessionAccess propagate without substitution.
    """
    eid = access.canonical_eid(eid)
    requests = tuple(recordings)
    if not requests:
        raise ValueError("Request at least one source recording")
    resolved = []
    seen = set()
    regions = BrainRegions()
    for index, request in enumerate(requests, 1):
        logger.info("neural-data %s: loading recording %d/%d (probe=%s, pid=%s)",
                    eid, index, len(requests), request.pname, request.pid)
        coverage = _coverage(request.coverage)
        result = access.load_spike_sorting(eid, pid=request.pid, pname=request.pname,
                                          collection=request.collection, revision=request.revision)
        source = result.source
        if source.eid != eid:
            raise ValueError("Spike source belongs to a different session")
        key = (source.pname, source.pid or "", source.collection, source.revision)
        if key in seen:
            raise ValueError("Duplicate recording/sorting request")
        seen.add(key)
        consumed_content = content_hash(dict(spikes=result.spikes, clusters=result.clusters,
                                             channels=result.channels))
        table = _unit_table(result.clusters, result.channels)
        times = _vector(result.spikes["times"], "spikes.times")
        assignments = _vector(result.spikes["clusters"], "spikes.clusters", len(times), integer=True)
        if times.dtype.kind not in "fiu" or not np.isfinite(times).all():
            raise ValueError("Spike times must be finite session-second values")
        if np.any(assignments < 0) or np.any(assignments >= len(table)):
            raise ValueError("Spike assignment does not resolve to a source unit")
        for name, values in result.spikes.items():
            array = np.asarray(values)
            if array.ndim == 0 or len(array) != len(times):
                raise ValueError(f"spikes.{name} does not match event rows")
        anatomical_mask, anatomical_policy = _anatomy(table, regions, anatomy)
        selected = anatomical_mask & _quality_mask(table, quality)
        original_ids = np.flatnonzero(selected)
        remap = np.full(len(table), -1, dtype=np.int64)
        remap[original_ids] = np.arange(len(original_ids))
        keep = selected[assignments]
        units = table.loc[selected].copy().reset_index(drop=True)
        for name, value in (("eid", eid), ("pid", source.pid), ("probe", source.pname),
                            ("collection", source.collection), ("revision", source.revision)):
            units[name] = value
        resolved.append((key, times[keep].copy(), remap[assignments[keep]], units, dict(
            source=asdict(source), content_sha256=consumed_content,
            request={k: v for k, v in asdict(request).items() if k != "coverage"},
            coverage=coverage, timing=dict(units="seconds", clock="source-session",
                provenance="SessionAccess.load_spike_sorting ALF times contract", conversion=None),
            selection=dict(quality=None if quality is None else asdict(quality), anatomy=anatomical_policy),
            source_unit_count=len(table), selected_unit_ids=original_ids.tolist(),
            status="available" if len(units) else "empty_selection")))
    resolved.sort(key=lambda item: item[0])
    times, assignments, unit_tables, records = [], [], [], []
    offset = 0
    for recording_index, (_, spike_times, spike_units, units, record) in enumerate(resolved):
        units["recording_index"] = recording_index
        times.append(spike_times)
        assignments.append(spike_units + offset)
        offset += len(units)
        unit_tables.append(units)
        records.append(record)
    times = np.concatenate(times)
    assignments = np.concatenate(assignments)
    order = np.argsort(times, kind="stable")
    return Population(eid, dict(times=times[order], clusters=assignments[order]),
                      pd.concat(unit_tables, ignore_index=True), tuple(records),
                      dict(quality=None if quality is None else asdict(quality),
                           anatomy=None if anatomy is None else asdict(anatomy)))
