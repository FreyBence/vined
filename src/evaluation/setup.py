"""Checkpoint restoration and scientific test-set compatibility checks."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

from loader.make_loader import make_loader
from trainer.artifacts import load_training_checkpoint, plain
from trainer.pretrained import model_from_checkpoint
from training_dataset.handoff import load_dataset_splits
from training_dataset import TARGET_SUPPORT_BINS
from training_dataset.context import STRICT_CONTEXT_BINS


@dataclass
class EvaluationSetup:
    model: object
    dataset: object
    dataloader: object
    configuration: dict
    provenance: dict

    def summary(self):
        return dict(self.provenance, evaluation_configuration=self.configuration,
                    test_sample_count=len(self.dataset), test_batches=len(self.dataloader))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _consumed_observations(checkpoint, path, visited=None, lineage=None):
    """Include previous optimization and selection observations in run ancestry."""
    visited = set() if visited is None else visited
    path = Path(path).resolve()
    if path in visited:
        raise ValueError("Cyclic checkpoint lineage")
    visited.add(path)
    if lineage is not None:
        lineage.append(dict(path=str(path), sha256=_sha256(path), run=checkpoint["run"],
                            epoch=checkpoint["epoch"], selection=checkpoint["selection"]))
    memberships = checkpoint["compatibility"].get("memberships")
    if not isinstance(memberships, dict) or "train" not in memberships:
        raise ValueError("Checkpoint lacks optimization/validation membership provenance")
    consumed = {(row["session_id"], row["trial_id"])
                for split in ("train", "val") for row in memberships.get(split, [])}
    run = checkpoint["run"]
    for name in ("parent", "pretrained"):
        reference = run.get(name)
        if not reference:
            continue
        source = Path(reference.get("checkpoint", reference.get("path", "")))
        if not source.is_file():
            raise ValueError(f"Cannot verify test isolation: missing {name} checkpoint {source}")
        if reference.get("sha256") and _sha256(source) != reference["sha256"]:
            raise ValueError("Pretrained checkpoint identity has changed")
        consumed.update(_consumed_observations(load_training_checkpoint(source), source, visited.copy(), lineage))
    return consumed


def resolve_setup(*, checkpoint_path, dataset_generation, context_mode, context_bins=None, expected_generation_id=None,
                  session_ids=None, batch_size=32, device="cpu", seed=42):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Inference batch size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Evaluation seed must be a nonnegative integer")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = load_training_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    context = config["training"].get("temporal_context")
    if (context_mode not in ("strict", "full_trial")
            or (context_mode == "strict" and (type(context_bins) is not int or context_bins not in STRICT_CONTEXT_BINS))
            or (context_mode == "full_trial" and context_bins is not None)):
        raise ValueError("Evaluation requires strict context with bins 1|3|6|9|12, or full_trial without bins")
    if (config["training"]["objective"] not in ("encoding", "decoding") or context is None
            or context.get("mode") != context_mode or context.get("bins") != context_bins
            or context.get("target_support_bins") != TARGET_SUPPORT_BINS
            or context.get("boundary_policy") != "complete"):
        raise ValueError("Requested evaluation context/support differs from the checkpoint; legacy checkpoints require a separate protocol")
    _, _, test, metadata = load_dataset_splits(
        dataset_generation, session_ids=session_ids, expected_generation_id=expected_generation_id)
    if not test:
        raise ValueError("Selected persisted test split is empty")
    # Generation identity binds timing, alignment, feature representation and split
    # configuration not fully described by checkpoint tensor dimensions.
    if metadata["dataset_generation_id"] != config["dataset"]["dataset_generation_id"]:
        raise ValueError("Evaluation requires the checkpoint's scientific dataset generation; "
                         "a different generation's temporal/representation compatibility is not established")
    strategy = metadata["dataset_metadata"]["configuration"]["strategy"]
    if strategy not in ("within_session", "session_held_out"):
        raise ValueError("Unsupported persisted evaluation strategy")
    rule = config["training"]["checkpoint_selection"]
    selection = checkpoint["selection"]
    if rule == "final":
        if checkpoint["next_epoch"] != config["training"]["num_epochs"]:
            raise ValueError("Final selection requires the configured final epoch")
    elif rule in ("validation_metric", "validation_loss"):
        if not selection or selection.get("rule") != rule or selection.get("epoch") != checkpoint["epoch"]:
            raise ValueError("Checkpoint weights do not correspond to the recorded selected epoch")
    else:
        raise ValueError("Unsupported checkpoint-selection policy")
    lineage = []
    consumed = _consumed_observations(checkpoint, checkpoint_path, lineage=lineage)
    if any((sample.session_id, sample.trial_id) in consumed for sample in test.samples):
        raise ValueError("Test observations overlap optimization or checkpoint-selection observations")
    populations = checkpoint["compatibility"]["populations"]
    max_time = config["data"]["max_time_length"]
    max_neurons = config["data"]["max_space_length"]
    width = config["model"]["encoder"]["embedder"].get("visual_dim", 768)
    for sample in test.samples:
        if sample.metadata["alignment"].get("visual_resampling") != "timestamp-aware held-state over neural intervals v1":
            raise ValueError("Temporal-context evaluation requires held-state alignment provenance")
        population = populations.get(sample.session_id)
        table = sample.neuron_identity
        if population is None:
            raise ValueError(f"No checkpoint output mapping for session {sample.session_id}; adapt before evaluation")
        if (population["columns"] != list(table.columns)
                or population["dtypes"] != [str(dtype) for dtype in table.dtypes]
                or population["rows"] != plain(table.to_dict(orient="records"))
                or config["dataset"]["eid_list"].get(sample.session_id) != sample.neuron_count):
            raise ValueError(f"Checkpoint ordered population differs for {sample.session_id}")
        if sample.visual.shape[1] != width or sample.sequence_length > max_time or sample.neuron_count > max_neurons:
            raise ValueError("Dataset feature or temporal/neural dimensions are incompatible with checkpoint")
    eligible = sum(max(0, sample.sequence_length - TARGET_SUPPORT_BINS + 1) for sample in test.samples)
    if not eligible:
        raise ValueError("Selected test split has no complete H=12 targets")
    model = model_from_checkpoint(checkpoint_path)
    if any(not torch.isfinite(value).all() for value in model.parameters()):
        raise ValueError("Checkpoint contains non-finite model parameters")
    identity = model.checkpoint_identity()
    if model.model_mode != config["training"]["objective"]:
        raise ValueError("Restored prediction direction differs from checkpoint configuration")
    if model.context_mode != context_mode or model.context_bins != context_bins:
        raise ValueError("Restored model context differs from the requested checkpoint context")
    if identity["neural_output"] != "log_expected_spike_count" or identity["visual_dim"] != width:
        raise ValueError("Restored model output/feature semantics differ from the dataset contract")
    requested_device = torch.device(device)
    model.to(requested_device).eval()
    loader = make_loader(test, batch_size=batch_size, target=["vision-clip"],
                         max_time_length=max_time, max_space_length=max_neurons,
                         pad_to_right=True, pad_value=-1., load_meta=config["data"]["load_meta"],
                         dataset_name=config["data"]["dataset_name"], stitching=True,
                         shuffle=False, seed=seed, mode="test", eids=metadata["eids"])
    effective = dict(batch_size=batch_size, device=str(requested_device), seed=seed,
                     temporal_context=plain(context),
                     target_coverage=dict(eligible_temporal_targets=eligible,
                         excluded_temporal_targets=sum(sample.sequence_length for sample in test.samples) - eligible,
                         trials_without_targets=sum(sample.sequence_length < TARGET_SUPPORT_BINS for sample in test.samples)),
                     model_mode=model.model_mode, modal_filter=config["training"]["modal_filter"],
                     split="test", session_ids=metadata["selected_session_ids"])
    provenance = dict(checkpoint_path=str(checkpoint_path), checkpoint_sha256=_sha256(checkpoint_path),
                      checkpoint_epoch=checkpoint["epoch"], checkpoint_selection_policy=rule,
                      checkpoint_selection=selection, training_run=checkpoint["run"],
                      checkpoint_lineage=lineage,
                      model_identity=identity, model_configuration=config["model"],
                      dataset_generation_id=metadata["dataset_generation_id"],
                      dataset_path=metadata["dataset_path"], evaluation_strategy=strategy,
                      dataset_metadata=metadata["dataset_metadata"])
    return EvaluationSetup(model, test, loader, effective, provenance)
