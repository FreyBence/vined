"""Verified split views and a compatible, scientifically explicit loader boundary."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.provenance import fingerprint
from .artifacts import load_dataset
from .samples import TrainingSample, _integer
from .splits import SPLITS


def _unit_column(sample, name, missing):
    table = sample.neuron_identity
    return table[name].tolist() if name in table else [missing] * sample.neuron_count


def _regions(sample):
    return [str(value) if pd.notna(value)
            else "" for value in _unit_column(sample, "acronym", "")]


def _unit_labels(sample):
    if "uuids" in sample.neuron_identity and sample.neuron_identity["uuids"].notna().all():
        return [str(value) for value in sample.neuron_identity["uuids"]]
    keys = ["eid", "probe", "collection", "revision", "source_unit_id", "recording_index"]
    return ["unit-" + fingerprint({key: None if pd.isna(value) else str(value)
                                   for key, value in zip(keys, row)})
            for row in sample.neuron_identity[keys].itertuples(index=False, name=None)]


@dataclass(frozen=True)
class PersistedSplit:
    """Samples selected from one verified generation, with legacy metadata columns."""

    samples: tuple[TrainingSample, ...]
    split: str
    generation_id: str
    session_ids: tuple[str, ...]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        if not isinstance(key, str):
            return self.samples[key]
        if key == "cluster_regions":
            return [_regions(sample) for sample in self.samples]
        if key == "cluster_uuids":
            return [_unit_labels(sample) for sample in self.samples]
        if key == "eid":
            return [sample.session_id for sample in self.samples]
        if key == "intervals":
            return [[sample.aligned_start, sample.aligned_end] for sample in self.samples]
        if key in ("sample_id", "trial_id", "sequence_length", "neuron_count", "split"):
            return [getattr(sample, key) for sample in self.samples]
        raise KeyError(f"Unsupported scientific split column: {key}")


def load_dataset_splits(path, *, session_ids=None, expected_generation_id=None):
    """Return train/val/test views and metadata; persisted memberships are authoritative."""
    generation = load_dataset(path, expected_generation_id=expected_generation_id)
    available = generation.manifest["sessions"]
    selected = available if session_ids is None else list(session_ids)
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(available):
        raise ValueError("Select unique sessions present in the dataset generation")
    selected = tuple(sorted(selected))
    views = tuple(PersistedSplit(tuple(sample for sample in getattr(generation.dataset, split)
                                     if sample.session_id in selected), split,
                                 generation.generation_id, selected) for split in SPLITS)
    counts = {sample.session_id: sample.neuron_count for view in views for sample in view.samples}
    metadata = dict(num_neurons=sorted(set(counts.values())), num_sessions=len(selected),
                    eids=list(selected), eid_list=dict(sorted(counts.items())),
                    dataset_generation_id=generation.generation_id, dataset_path=str(generation.path),
                    selected_session_ids=list(selected), dataset_metadata=generation.dataset.metadata)
    return (*views, metadata)


def model_sample(sample, *, max_time_length, max_space_length, pad_value=-1.):
    """Expose legacy tensor names plus scientific fields without dropping real data."""
    if not isinstance(sample, TrainingSample):
        raise TypeError("Model sample adaptation requires TrainingSample")
    time_size = _integer(max_time_length, "max_time_length", minimum=1)
    neuron_size = _integer(max_space_length, "max_space_length", minimum=1)
    length, neurons = sample.sequence_length, sample.neuron_count
    if length > time_size or neurons > neuron_size:
        raise ValueError("Loader maxima cannot truncate scientific time bins or neurons")
    if not np.isfinite(pad_value) or not np.isfinite(np.float32(pad_value)):
        raise ValueError("Loader padding value must be finite float32")
    counts = sample.neural[:length, :neurons]
    converted = counts.astype(np.float32)
    if counts.max() > 2**24 and any(int(a) != int(b) for a, b in zip(counts.flat, converted.flat)):
        raise ValueError("Spike counts cannot be represented exactly by the legacy float32 model input")
    neural = np.zeros((time_size, neuron_size), dtype=np.int64)
    neural[:length, :neurons] = counts
    spikes = np.full((time_size, neuron_size), pad_value, dtype=np.float32)
    spikes[:length, :neurons] = converted
    vision = np.full((time_size, sample.visual.shape[1]), pad_value, dtype=np.float32)
    vision[:length] = sample.visual[:length]
    time_mask = np.arange(time_size) < length
    neuron_mask = np.arange(neuron_size) < neurons
    physical = {}
    for name in ("physical_timestamps", "bin_start_times", "bin_end_times"):
        values = np.full(time_size, np.nan, dtype=np.float64)
        values[:length] = getattr(sample, name)[:length]
        physical[name] = values
    depths = np.full(neuron_size, np.nan, dtype=np.float32)
    depths[:neurons] = np.asarray([value if pd.notna(value) else np.nan
                                  for value in _unit_column(sample, "depths", np.nan)], dtype=np.float32)
    positions = np.arange(time_size, dtype=np.int64)
    temporal_positions = positions.copy()
    temporal_positions[~time_mask] = -1
    visual = np.zeros_like(vision)
    visual[:length] = sample.visual[:length]
    return dict(spikes_data=spikes, neural=neural, visual=visual, **{"vision-clip": vision,
        "vision-clip_valid": time_mask.copy()}, time_attn_mask=time_mask.astype(np.int64),
        space_attn_mask=neuron_mask.astype(np.int64), temporal_mask=time_mask, neuron_mask=neuron_mask,
        spikes_timestamps=positions, temporal_positions=temporal_positions,
        spikes_spacestamps=np.arange(neuron_size, dtype=np.int64), **physical,
        neuron_depths=depths, neuron_regions=_regions(sample) + [""] * (neuron_size - neurons),
        neuron_identity=sample.neuron_identity.copy(deep=True), metadata=sample.metadata,
        sample_id=sample.sample_id, eid=sample.session_id, session_id=sample.session_id,
        trial_id=sample.trial_id, split=sample.split, sequence_length=length, neuron_count=neurons,
        intervals=np.array([sample.aligned_start, sample.aligned_end], dtype=np.float64),
        stim_on=sample.stim_on, stim_off=sample.stim_off, aligned_start=sample.aligned_start,
        aligned_end=sample.aligned_end, bin_size=sample.bin_size,
        discarded_tail_duration=sample.discarded_tail_duration)


def collate_model_samples(rows):
    """Use standard tensor collation while retaining variable scientific metadata per sample."""
    from torch.utils.data import default_collate
    sidecars = ("neuron_identity", "metadata")
    batch = default_collate([{key: value for key, value in row.items() if key not in sidecars} for row in rows])
    batch.update({key: [row[key] for row in rows] for key in sidecars})
    return batch
