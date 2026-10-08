"""Complete inference collection with source identity and explicit validity."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from evaluation.setup import EvaluationSetup


@dataclass(frozen=True)
class EvaluationPrediction:
    sample_id: str
    session_id: str
    trial_id: int
    neuron_identity: pd.DataFrame
    temporal_positions: np.ndarray
    physical_timestamps: np.ndarray
    bin_start_times: np.ndarray
    bin_end_times: np.ndarray
    bin_size: float
    stim_on: float
    stim_off: float
    aligned_start: float
    aligned_end: float
    temporal_mask: np.ndarray
    neuron_mask: np.ndarray
    observed_neural: np.ndarray
    observed_visual: np.ndarray
    predicted_neural: np.ndarray | None
    predicted_visual: np.ndarray | None
    metadata: dict


@dataclass(frozen=True)
class PredictionCollection:
    predictions: tuple[EvaluationPrediction, ...]
    configuration: dict
    provenance: dict

    def summary(self):
        sessions = {}
        for row in self.predictions:
            entry = sessions.setdefault(row.session_id, dict(trials=0, neurons=len(row.neuron_identity)))
            entry["trials"] += 1
        return dict(sample_count=len(self.predictions), sessions=sessions,
                    prediction_direction=self.configuration["model_mode"],
                    checkpoint_sha256=self.provenance["checkpoint_sha256"],
                    dataset_generation_id=self.provenance["dataset_generation_id"],
                    neural_output="log_expected_spike_count", metrics_computed=False)


def _dictionary_prediction(model, batch, direction, device):
    target = "spike" if direction == "encoding" else "vision-clip"
    data = {}
    for mod in model.encoder_embeddings:
        values = batch["spikes_data" if mod == "spike" else "vision-clip"].to(device)
        temporal = batch["temporal_mask"].to(device)
        data[mod] = dict(inputs=values, eid=list(batch["session_id"]),
                         inputs_modality=torch.tensor(model.mod_to_indx[mod], device=device),
                         inputs_timestamp=batch["temporal_positions"].to(device),
                         temporal_mask=temporal,
                         training_mask=temporal if mod == target else torch.zeros_like(temporal))
        if mod == "spike":
            data[mod]["neuron_mask"] = batch["neuron_mask"].to(device)
    output = model(data)
    if not torch.equal(output.temporal_mask.cpu(), batch["temporal_mask"]):
        raise ValueError("Model output temporal validity differs from scientific batch validity")
    if target == "spike" and not torch.equal(output.neuron_mask.cpu(), batch["neuron_mask"]):
        raise ValueError("Model output neuron validity differs from scientific batch validity")
    return output.mod_preds[target]


def _infer(model, batch, device):
    if model.model_mode == "encoding":
        output = model(visual_features=batch["vision-clip"].to(device),
                       temporal_positions=batch["temporal_positions"].to(device),
                       temporal_mask=batch["temporal_mask"].to(device),
                       neuron_mask=batch["neuron_mask"].to(device),
                       session_id=list(batch["session_id"]))
        if (not torch.equal(output.temporal_mask.cpu(), batch["temporal_mask"])
                or not torch.equal(output.neuron_mask.cpu(), batch["neuron_mask"])):
            raise ValueError("Model output validity differs from scientific batch validity")
        return output.neural_prediction, None
    if model.model_mode == "decoding":
        return None, _dictionary_prediction(model, batch, "decoding", device)
    if model.model_mode == "mm":
        return (_dictionary_prediction(model, batch, "encoding", device),
                _dictionary_prediction(model, batch, "decoding", device))
    raise ValueError(f"Unsupported checkpoint direction: {model.model_mode}")


def collect_predictions(setup: EvaluationSetup) -> PredictionCollection:
    """Infer every test sample once; return detached CPU scientific records."""
    model = setup.model
    model.eval()
    device = next(model.parameters()).device
    expected = setup.dataset.samples
    records, seen = [], set()
    with torch.inference_mode():
        for batch in setup.dataloader:
            batch_size = len(batch["sample_id"])
            offset = len(records)
            for index, sample_id in enumerate(batch["sample_id"]):
                if sample_id in seen or offset + index >= len(expected):
                    raise ValueError("Duplicate or unexpected test sample in inference loader")
                source = expected[offset + index]
                if (sample_id != source.sample_id or batch["session_id"][index] != source.session_id
                        or int(batch["trial_id"][index]) != source.trial_id or batch["split"][index] != "test"):
                    raise ValueError("Inference loader identity/order differs from selected test split")
                length, neurons = source.sequence_length, source.neuron_count
                temporal = batch["temporal_mask"][index]
                neural = batch["neuron_mask"][index]
                if (not torch.equal(temporal, torch.arange(len(temporal)) < length)
                        or not torch.equal(neural, torch.arange(len(neural)) < neurons)
                        or not batch["neuron_identity"][index].equals(source.neuron_identity)):
                    raise ValueError(f"Inference validity/population differs for sample {sample_id}")
                if (not np.array_equal(batch["neural"][index, :length, :neurons].numpy(),
                                       source.neural[:length, :neurons])
                        or not np.array_equal(batch["spikes_data"][index, :length, :neurons].numpy(),
                                              source.neural[:length, :neurons])
                        or not np.array_equal(batch["visual"][index, :length].numpy(), source.visual[:length])
                        or not np.array_equal(batch["vision-clip"][index, :length].numpy(), source.visual[:length])
                        or not torch.equal(batch["vision-clip_valid"][index], temporal)):
                    raise ValueError(f"Inference targets differ from scientific sample {sample_id}")
                for key in ("temporal_positions", "physical_timestamps", "bin_start_times", "bin_end_times"):
                    if not np.array_equal(batch[key][index, :length].numpy(), getattr(source, key)[:length]):
                        raise ValueError(f"Inference {key} differs for scientific sample {sample_id}")
                if float(batch["bin_size"][index]) != source.bin_size:
                    raise ValueError(f"Inference bin duration differs for scientific sample {sample_id}")
                seen.add(sample_id)
            neural_prediction, visual_prediction = _infer(model, batch, device)
            for name, prediction, shape in (
                    ("neural", neural_prediction, batch["neural"].shape),
                    ("visual", visual_prediction, batch["visual"].shape)):
                if prediction is not None and prediction.shape != shape:
                    raise ValueError(f"Incompatible {name} prediction shape for batch at sample {offset}")
            for index in range(batch_size):
                source = expected[offset + index]
                def array(key):
                    return batch[key][index].detach().cpu().numpy().copy()
                records.append(EvaluationPrediction(
                    sample_id=source.sample_id, session_id=source.session_id, trial_id=source.trial_id,
                    neuron_identity=source.neuron_identity.copy(deep=True),
                    temporal_positions=array("temporal_positions"),
                    physical_timestamps=array("physical_timestamps"),
                    bin_start_times=array("bin_start_times"), bin_end_times=array("bin_end_times"),
                    bin_size=source.bin_size, stim_on=source.stim_on, stim_off=source.stim_off,
                    aligned_start=source.aligned_start, aligned_end=source.aligned_end,
                    temporal_mask=array("temporal_mask"),
                    neuron_mask=array("neuron_mask"), observed_neural=array("neural"),
                    observed_visual=array("visual"),
                    predicted_neural=None if neural_prediction is None else neural_prediction[index].cpu().numpy().copy(),
                    predicted_visual=None if visual_prediction is None else visual_prediction[index].cpu().numpy().copy(),
                    metadata=deepcopy(source.metadata)))
    if len(records) != len(expected):
        raise ValueError(f"Incomplete test coverage: collected {len(records)} of {len(expected)} samples")
    return PredictionCollection(tuple(records), deepcopy(setup.configuration), deepcopy(setup.provenance))
