"""Identity-based dataset splits with explicit, serializable membership metadata."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import floor, isclose, isfinite
from numbers import Real
from typing import Iterable, Mapping, Optional
from uuid import UUID

from utils.provenance import fingerprint
from .samples import TrainingSample, _integer

SPLITS = ("train", "val", "test")


def _session(value):
    if not isinstance(value, str):
        raise ValueError("Session identities must be canonical UUID strings")
    if str(UUID(value)) != value:
        raise ValueError("Session identities must be canonical UUID strings")
    return value


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    ratios: Optional[tuple[float, float, float]] = None
    seed: Optional[int] = None
    session_assignments: Optional[Mapping[str, str]] = None
    exclusions: Mapping[tuple[str, int], str] = field(default_factory=dict)

    def __post_init__(self):
        if self.strategy not in ("within_session", "session_held_out"):
            raise ValueError("strategy must be within_session or session_held_out")
        explicit = self.session_assignments is not None
        if explicit:
            if self.strategy != "session_held_out" or self.ratios is not None or self.seed is not None:
                raise ValueError("Explicit session assignments require session_held_out without ratios/seed")
            if not isinstance(self.session_assignments, Mapping) or not self.session_assignments:
                raise ValueError("session_assignments must be a nonempty session-to-split mapping")
            assignments = {_session(eid): split for eid, split in self.session_assignments.items()}
            if any(split not in SPLITS for split in assignments.values()):
                raise ValueError("Session assignments must use train, val, or test")
            object.__setattr__(self, "session_assignments", assignments)
        else:
            if self.ratios is None or len(self.ratios) != 3:
                raise ValueError("Provide train/val/test ratios or explicit held-out session assignments")
            if any(isinstance(ratio, bool) or not isinstance(ratio, Real)
                   or not isfinite(ratio) or not 0 <= ratio <= 1 for ratio in self.ratios):
                raise ValueError("Split ratios must be finite numbers in [0,1]")
            ratios = tuple(float(ratio) for ratio in self.ratios)
            if not isclose(sum(ratios), 1., rel_tol=0, abs_tol=1e-12):
                raise ValueError("Split ratios must sum to one")
            object.__setattr__(self, "ratios", ratios)
            object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        if not isinstance(self.exclusions, Mapping):
            raise ValueError("exclusions must map source trial identities to reasons")
        exclusions = {}
        for identity, reason in self.exclusions.items():
            if not isinstance(identity, tuple) or len(identity) != 2:
                raise ValueError("Exclusion identities must be (session_id, trial_id)")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("Every exclusion requires a nonempty reason")
            exclusions[(_session(identity[0]), _integer(identity[1], "trial_id"))] = reason
        object.__setattr__(self, "exclusions", exclusions)


@dataclass(frozen=True)
class DatasetSplits:
    train: tuple[TrainingSample, ...]
    val: tuple[TrainingSample, ...]
    test: tuple[TrainingSample, ...]
    metadata: dict


def _allocate(groups, config):
    """Rank source identities, then apportion group counts with stable tie-breaking."""
    count = len(groups)
    positive = [index for index, ratio in enumerate(config.ratios) if ratio > 0]
    if count < len(positive):
        raise ValueError("Not enough independent groups to populate every positive-ratio split")
    quotas = [count * ratio for ratio in config.ratios]
    sizes = [floor(quota) for quota in quotas]
    remainder_order = sorted(range(3), key=lambda index: (-(quotas[index] - sizes[index]), index))
    for index in remainder_order[:count - sum(sizes)]:
        sizes[index] += 1
    for index in positive:
        if sizes[index] == 0:
            donor = max((i for i in positive if sizes[i] > 1), key=lambda i: (sizes[i], -i))
            sizes[donor] -= 1
            sizes[index] = 1
    ranked = sorted(groups, key=lambda identity: (
        fingerprint(dict(strategy=config.strategy, seed=config.seed, group=identity)), identity))
    memberships, start = {}, 0
    for split, size in zip(SPLITS, sizes):
        memberships.update({identity: split for identity in ranked[start:start + size]})
        start += size
    return memberships


def _record(sample):
    return dict(sample_id=sample.sample_id, session_id=sample.session_id, trial_id=int(sample.trial_id))


def assign_splits(samples: Iterable[TrainingSample], *, config: SplitConfig) -> DatasetSplits:
    """Assign whole source trials or sessions; enumeration and padding cannot affect grouping."""
    if not isinstance(config, SplitConfig):
        raise TypeError("config must be SplitConfig")
    # Revalidate and snapshot caller-owned mapping fields before using them.
    config = SplitConfig(config.strategy, config.ratios, config.seed,
                         config.session_assignments, config.exclusions)
    samples = tuple(samples)
    if not samples:
        raise ValueError("Split assignment requires at least one sample")
    sample_ids, source_groups = set(), defaultdict(list)
    for sample in samples:
        if not isinstance(sample, TrainingSample):
            raise TypeError("Split assignment requires TrainingSample objects")
        identity = (_session(sample.session_id), _integer(sample.trial_id, "trial_id"))
        if not isinstance(sample.sample_id, str) or not sample.sample_id or sample.sample_id in sample_ids:
            raise ValueError("Samples must have unique nonempty sample IDs")
        if sample.split is not None:
            raise ValueError("Supply unassigned samples rather than overwriting existing memberships")
        sample_ids.add(sample.sample_id)
        source_groups[identity].append(sample)
    unknown = set(config.exclusions) - set(source_groups)
    if unknown:
        raise ValueError(f"Exclusions refer to absent source trials: {sorted(unknown)}")
    retained = sorted(set(source_groups) - set(config.exclusions))
    if not retained:
        raise ValueError("Exclusions leave no samples to split")
    sessions = sorted({eid for eid, _ in retained})
    assignment = {}
    if config.strategy == "within_session":
        for eid in sessions:
            assignment.update(_allocate([identity for identity in retained if identity[0] == eid], config))
    else:
        if config.session_assignments is not None:
            if set(config.session_assignments) != set(sessions):
                raise ValueError("Session assignments must name exactly the retained sessions")
            session_splits = config.session_assignments
        else:
            session_splits = _allocate(sessions, config)
        assignment = {identity: session_splits[identity[0]] for identity in retained}
    configuration = dict(strategy=config.strategy, ratios=config.ratios, seed=config.seed,
                         session_assignments=(dict(sorted(config.session_assignments.items()))
                                              if config.session_assignments is not None else None),
                         exclusions=[dict(session_id=eid, trial_id=trial_id, reason=reason)
                                     for (eid, trial_id), reason in sorted(config.exclusions.items())])
    configuration_id = fingerprint(configuration)
    result = {split: [] for split in SPLITS}
    for identity in retained:
        split = assignment[identity]
        for sample in sorted(source_groups[identity], key=lambda item: item.sample_id):
            metadata = deepcopy(sample.metadata)
            metadata["split"] = dict(name=split, configuration_id=configuration_id)
            result[split].append(replace(sample, split=split, metadata=metadata))
    metadata = dict(configuration=configuration, configuration_id=configuration_id,
                    algorithm="sha256-ranked-largest-remainder-nonempty-v1"
                              if config.ratios is not None else "explicit-session-assignments-v1",
                    grouping="source_trial" if config.strategy == "within_session" else "session",
                    memberships={split: [_record(sample) for sample in result[split]] for split in SPLITS},
                    exclusions=[dict(_record(sample), reason=config.exclusions[identity])
                                for identity in sorted(config.exclusions)
                                for sample in sorted(source_groups[identity], key=lambda item: item.sample_id)])
    return DatasetSplits(tuple(result["train"]), tuple(result["val"]), tuple(result["test"]), metadata)
