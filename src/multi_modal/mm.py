import os
import copy
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from transformers.activations import ACT2FN

ACT2FN["softsign"] = nn.Softsign
from models.masker import Masker
from models.model_output import ModelOutput
from models.stitcher import StitchDecoder, session_populations, session_indices
from multi_modal.encoder_embeddings import EncoderLayer
from multi_modal.mm_utils import create_context_mask
from utils.config_utils import DictConfig, update_config


STATIC_VARS = []
DYNAMIC_VARS = ["vision-clip"]

@dataclass
class MultiModalOutput(ModelOutput):
    neural_prediction: Optional[torch.FloatTensor] = None
    temporal_mask: Optional[torch.BoolTensor] = None
    neuron_mask: Optional[torch.BoolTensor] = None
    latent_representation: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    mod_loss: Optional[Dict[str, torch.Tensor]] = None
    mod_n_examples: Optional[Dict[str, torch.Tensor]] = None
    mod_preds: Optional[Dict[str, torch.Tensor]] = None
    mod_targets: Optional[Dict[str, torch.Tensor]] = None
    static_preds: Optional[torch.LongTensor] = None
    static_targets: Optional[torch.LongTensor] = None
    
class MultiModal(nn.Module):
    def __init__(
        self, 
        encoder_embeddings:        Dict[str, nn.Module],
        avail_mod:                 List,
        avail_beh:                 List,
        model_mode:                List,
        config:                    DictConfig,
        neuron_order:              Optional[Dict[str, List]] = None,
        **kwargs
    ):
        super().__init__()

        self.avail_mod = avail_mod
        self.avail_beh = avail_beh
        self.model_mode = model_mode
        self.eid_list = session_populations(kwargs["eid_list"])
        self.neuron_order = copy.deepcopy(neuron_order)
        if self.neuron_order is not None:
            if (not isinstance(self.neuron_order, dict)
                    or set(self.neuron_order) != set(self.eid_list)
                    or any(not isinstance(self.neuron_order[session], list)
                           or len(self.neuron_order[session]) != count
                           for session, count in self.eid_list.items())):
                raise ValueError("neuron_order must contain one ordered list per session population")
        self.architecture_config = copy.deepcopy(dict(config))
        self.architecture_config.pop("masker", None)
        self.mod_to_indx = {r: i for i, r in enumerate(self.avail_mod)}
        self.n_mod = len(self.avail_mod)

        self.n_layers = config.encoder.transformer.n_layers
        self.hidden_size = config.encoder.transformer.hidden_size
        self.max_F = config.encoder.embedder.max_F
        self.visual_dim = config.encoder.embedder.get("visual_dim", 768)
        if (not isinstance(self.max_F, int) or isinstance(self.max_F, bool) or self.max_F <= 0
                or not isinstance(self.visual_dim, int) or isinstance(self.visual_dim, bool) or self.visual_dim <= 0):
            raise ValueError("max_F and visual_dim must be positive integers")
        context = config.get("context", {})
        self.context_forward = context.get("forward", -1)
        self.context_backward = context.get("backward", -1)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < -1
               for v in (self.context_forward, self.context_backward)):
            raise ValueError("Context limits must be -1 or nonnegative integer positions")

        self.encoder_modalities = set(encoder_embeddings.keys())
        self.encoder_embeddings = nn.ModuleDict(encoder_embeddings)
        for embedding in self.encoder_embeddings.values():
            if embedding.embedder.eid_list != self.eid_list:
                raise ValueError("Encoder and model session populations must agree")
            if embedding.max_F != self.max_F or embedding.visual_dim != self.visual_dim:
                raise ValueError("Encoder dimensions and maximum length must match model configuration")

        self.mask = config.masker.force_active
        if self.mask:
            assert config.masker.mode in ["temporal", "causal"], "only allow temporal / causal token masking."
            self.masker = Masker(config.masker)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        self.encoder = nn.ModuleList(
            [EncoderLayer(idx, config.encoder.transformer, self.max_F) for idx in range(self.n_layers)]
        )
        self.encoder_norm = nn.LayerNorm(self.hidden_size) 

        self.num_class = {
            "spike": None, "vision-clip": self.visual_dim,
        }
        self.mod_type = {
            "spike": "spike", 
            "vision-clip": "dynamic", 
        }

        self.mod_loss = {
            "spike": nn.PoissonNLLLoss(
                reduction="none",
                log_input=True
            ),

            "dynamic": self.cosine_loss,

            "static": nn.CrossEntropyLoss(reduction="none")
        }

        if self.model_mode in ["encoding", "decoding"]:
            self.init_unimodal_stitcher()
        
    def cosine_loss(self, preds, targets):

        preds = F.normalize(preds, dim=-1)
        targets = F.normalize(targets, dim=-1)

        loss = 1 - (preds * targets).sum(dim=-1)

        return loss

    def init_unimodal_stitcher(self):
        # Trick to handle incompatibility between unimodal and multimodal outputs
        # Can we improve this in the future?
        if self.model_mode == "encoding":
            mod_list = ["spike"]
            _eid_list = {k: v for k, v in self.eid_list.items()}
            n_channels = self.hidden_size * len(self.avail_beh)
        else:
            mod_list, _eid_list, n_channels = self.avail_beh, self.eid_list, self.hidden_size
            
        self.mod_stitcher_proj_dict = nn.ModuleDict()
        self.mod_static_weight_dict = nn.ModuleDict()
        self.mod_token_weight_dict = {}
        for mod in mod_list:
            self.mod_stitcher_proj_dict[mod] = StitchDecoder(
                eid_list = _eid_list, n_channels = n_channels, mod = mod,
                max_F=self.max_F, visual_dim=self.visual_dim,
            )
            if mod in STATIC_VARS:
                tmp_dict = {}
                for key, val in _eid_list.items():
                    tmp_dict[str(key)] = nn.Parameter(torch.rand(self.max_F))
                self.mod_static_weight_dict[mod] = nn.ParameterDict(tmp_dict)

    def checkpoint_identity(self):
        """Identity to persist with weights, independent of training artifacts."""
        return copy.deepcopy({
            "version": 1,
            "architecture": self.architecture_config,
            "model_mode": self.model_mode,
            "avail_mod": list(self.avail_mod),
            "avail_beh": list(self.avail_beh),
            "session_populations": self.eid_list,
            "session_embedding_ids": {
                mod: list(embedding.embedder.eid_lookup)
                for mod, embedding in self.encoder_embeddings.items()
            },
            "encoder_layout": {
                mod: {"input_channels": embedding.n_channel,
                      "output_channels": embedding.output_channel,
                      "max_F": embedding.max_F,
                      "stitching": hasattr(embedding.embedder, "mod_stitch_encoder")}
                for mod, embedding in self.encoder_embeddings.items()
            },
            "neuron_columns": {session: list(range(count)) for session, count in self.eid_list.items()},
            "ordered_units": self.neuron_order,
            "neural_output": "log_expected_spike_count",
            "neural_head_layout": "real_session_channels_then_padding",
            "visual_dim": self.visual_dim,
            "temporal_context": {"forward": self.context_forward, "backward": self.context_backward},
        })

    def validate_checkpoint_identity(self, identity):
        """Reject dimension-only compatibility; adaptation must be intentional."""
        if identity != self.checkpoint_identity():
            raise ValueError("Incompatible model/session identity; initialize adapted components explicitly")

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # Keep tensor entries unchanged for the existing training checkpoint consumers.
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]]["model_identity"] = self.checkpoint_identity()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        identity = local_metadata.get("model_identity")
        if identity is None:
            raise ValueError("Checkpoint lacks model identity; historical session row order cannot be inferred")
        self.validate_checkpoint_identity(identity)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
                
    
    def cat_encoder_tensors(self, mod_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        encoder_tokens, encoder_emb, input_timestamp = [], [], []
        encoder_mask, mod_mask = [], []

        for mod, d in mod_dict.items():
            encoder_tokens.append(d["x"])
            encoder_emb.append(d["emb"])
            input_timestamp.append(d["inputs_timestamp"])
            encoder_mask.append(d["inputs_mask"])
            mod_mask.append(torch.full_like(d["inputs_mask"], self.mod_to_indx[mod]))
    
        encoder_tokens = torch.cat(encoder_tokens, dim=1)
        encoder_emb = torch.cat(encoder_emb, dim=1)
        input_timestamp = torch.cat(input_timestamp, dim=1)
        encoder_mask = torch.cat(encoder_mask, dim=1)
        mod_mask = torch.cat(mod_mask, dim=1).to(torch.int16)
        return encoder_tokens, encoder_emb, input_timestamp, encoder_mask, mod_mask

    
    def forward_mask_encoder(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor]:
        
        encoder_tokens, encoder_emb, input_timestamp, encoder_mask, mod_mask = \
        self.cat_encoder_tensors(mod_dict)

        # encoder_tokens: [B, N, D]
        # encoder_mask:   [B, N], where 1 means masked
    
        mask = encoder_mask.unsqueeze(-1).bool()   # [B, N, 1]
    
        encoder_tokens = torch.where(
            mask,
            self.mask_token.expand_as(encoder_tokens),
            encoder_tokens,
        )       
        return encoder_tokens, encoder_emb, input_timestamp, encoder_mask, mod_mask


    def forward_encoder(
        self, 
        x: torch.Tensor, 
        input_timestamp: Optional[torch.LongTensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        B, N, _ = x.size()
        if temporal_mask is None:
            temporal_mask = torch.ones((B, N), device=x.device, dtype=torch.bool)
        temporal_mask = self._valid_mask(temporal_mask, (B, N), x.device, "temporal_mask")
        x = torch.where(temporal_mask.unsqueeze(-1), x, torch.zeros_like(x))
        attention_keys = temporal_mask.clone()
        # Empty samples have no scientific outputs. Supply one zero key to avoid
        # undefined all-masked attention on numerical backends, then zero outputs.
        attention_keys[~attention_keys.any(dim=1), 0] = True
        attention_mask = attention_keys[:, None, :]
        if input_timestamp is None:
            raise ValueError("Temporal processing requires explicit model positions")
        if self.context_forward != -1 or self.context_backward != -1:
            context = create_context_mask(self.context_forward, self.context_backward,
                                          self.max_F, input_timestamp)
            context = context | ~temporal_mask[:, :, None]
            attention_mask = attention_mask & context
        
        for layer in self.encoder:
            x = layer(
                x, mask=attention_mask, timestamp=input_timestamp
            )
            x = torch.where(temporal_mask.unsqueeze(-1), x, torch.zeros_like(x))

        x = self.encoder_norm(x)

        return torch.where(temporal_mask.unsqueeze(-1), x, torch.zeros_like(x))

    @staticmethod
    def _valid_mask(mask, shape, device, name):
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)}")
        if mask.device != device or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{name} must contain boolean/0-or-1 values on the input device")
        return mask.bool()

    def _neuron_mask(self, eid, batch_size, device, mask=None):
        session_indices(eid, self.eid_list, batch_size, device)
        width = max(self.eid_list.values())
        counts = torch.tensor([self.eid_list[session] for session in eid], device=device)
        real = torch.arange(width, device=device)[None, :] < counts[:, None]
        if mask is None:
            return real
        mask = self._valid_mask(mask, real.shape, device, "neuron_mask")
        if torch.any(mask & ~real):
            raise ValueError("neuron_mask marks session padding as real neurons")
        return mask

    def encode(self, visual_features, temporal_positions, temporal_mask, session_id, *,
               training_mask=None, neural_targets=None, neuron_mask=None,
               target_mask=None, return_latent=False):
        """Predict log spike counts without requiring targets or training masking."""
        if self.model_mode != "encoding" or set(self.encoder_embeddings) != {"vision-clip"}:
            raise ValueError("Visual-only prediction requires a visual-only encoding model")
        if not isinstance(visual_features, torch.Tensor) or visual_features.ndim != 3:
            raise ValueError("visual_features must have shape [B,T,D_visual]")
        B, T, _ = visual_features.shape
        if B == 0 or not 0 < T <= self.max_F or visual_features.size(-1) != self.visual_dim:
            raise ValueError("Encoding requires configured visual width and 0 < T <= max_F")
        device = visual_features.device
        valid = self._valid_mask(temporal_mask, (B, T), device, "temporal_mask")
        neurons = self._neuron_mask(session_id, B, device, neuron_mask)
        if (not isinstance(temporal_positions, torch.Tensor)
                or temporal_positions.shape != valid.shape or temporal_positions.device != device
                or temporal_positions.dtype != torch.long):
            raise ValueError("temporal_positions must be int64 [B,T] on the input device")
        corruption = torch.zeros_like(valid) if training_mask is None else self._valid_mask(
            training_mask, valid.shape, device, "training_mask") & valid
        data = {
            "inputs": torch.where((valid & ~corruption).unsqueeze(-1), visual_features,
                                  torch.zeros_like(visual_features)),
            "inputs_timestamp": torch.where(valid, temporal_positions, torch.zeros_like(temporal_positions)),
            "inputs_modality": torch.tensor(self.mod_to_indx["vision-clip"], device=device),
            "inputs_mask": corruption,
            "eid": session_id,
        }
        embedding = self.encoder_embeddings["vision-clip"](data)
        tokens, context, positions, _, _ = self.forward_mask_encoder({"vision-clip": embedding})
        latent = self.forward_encoder(tokens + context, positions, valid)
        if "vision-clip" in self.avail_beh:
            latent = F.normalize(latent, dim=-1)
        predictions = self.mod_stitcher_proj_dict["spike"](latent, session_id)
        cell_valid = valid.unsqueeze(-1) & neurons.unsqueeze(1)
        predictions = torch.where(cell_valid, predictions, torch.zeros_like(predictions))
        output = MultiModalOutput(neural_prediction=predictions, mod_preds={"spike": predictions},
                                  temporal_mask=valid, neuron_mask=neurons,
                                  latent_representation=latent if return_latent else None)
        if neural_targets is not None:
            selected = valid if target_mask is None else target_mask
            loss_data = {"spike": {"preds": predictions, "gt": neural_targets,
                                    "targets_mask": selected, "temporal_mask": valid,
                                    "neuron_mask": neurons}}
            (output.loss, output.mod_loss, output.mod_n_examples, output.mod_preds,
             output.mod_targets, output.static_targets, output.static_preds) = self.forward_loss(loss_data)
        elif target_mask is not None:
            raise ValueError("target_mask requires neural_targets")
        return output

    
    def forward_loss(self, 
        output_mod_dict: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        mod_loss, mod_n_examples, mod_preds, mod_targets, static_targets, static_preds = \
        {}, {}, {}, {}, {}, {}

        for mod, d in output_mod_dict.items():
            mod_preds[mod] = d["preds"]
            if "gt" not in d:
                continue
            targets = output_mod_dict[mod]["gt"]
            if not isinstance(targets, torch.Tensor) or targets.ndim != 3:
                raise ValueError(f"{mod} targets must have shape [B,T,D]")
            B, T, N = targets.size()
            targets_mask = output_mod_dict[mod]["targets_mask"]
            if not isinstance(targets_mask, torch.Tensor):
                raise ValueError("targets_mask must be a tensor")
            preds = output_mod_dict[mod]["preds"]

            mod_type = self.mod_type[mod]
            if mod_type != "static":
                if preds.shape != targets.shape or preds.device != targets.device:
                    raise ValueError(f"{mod} predictions and targets must have matching shapes/devices")
                temporal = self._valid_mask(d["temporal_mask"], (B, T), preds.device, "temporal_mask")
                if mod == "spike":
                    neurons = self._valid_mask(d["neuron_mask"], (B, N), preds.device, "neuron_mask")
                    if targets_mask.shape == (B, T):
                        targets_mask = self._valid_mask(targets_mask, (B, T), preds.device,
                                                       "targets_mask").unsqueeze(-1).expand(B, T, N)
                    selected = self._valid_mask(targets_mask, (B, T, N), preds.device, "targets_mask")
                    selected = selected & temporal.unsqueeze(-1) & neurons.unsqueeze(1)
                else:
                    selected = self._valid_mask(targets_mask, (B, T), preds.device, "targets_mask") & temporal
                n_examples = selected.sum()
                # Slice before numerical operations: invalid NaN/Inf targets must
                # never enter normalization, Poisson exponentiation, or gradients.
                selected_preds, selected_targets = preds[selected], targets[selected]
                if not torch.isfinite(selected_targets).all() or not torch.isfinite(selected_preds).all():
                    raise ValueError(f"Non-finite valid {mod} targets or predictions")
                if mod == "spike" and torch.any(selected_targets < 0):
                    raise ValueError("Valid spike targets must be nonnegative counts")
                loss = self.mod_loss[mod_type](selected_preds, selected_targets).sum() / n_examples.clamp_min(1)
            else:
                preds, targets = preds.squeeze(1), targets.squeeze(1)
                targets_mask = targets_mask.squeeze(1)
                n_examples = targets_mask.sum()
                if n_examples == 0:
                    mod_loss[mod], mod_n_examples[mod], mod_preds[mod], mod_targets[mod] = \
                    torch.zeros(1, device=targets.device, requires_grad=True).squeeze(), \
                    n_examples, preds, targets
                    continue                
                static_targets[mod], static_preds[mod] = targets.squeeze(1), preds.argmax(-1) 
                targets = targets.reshape(-1, 1)
                loss = self.mod_loss[mod_type](preds.float(), targets.float()).sum() / n_examples
            
            mod_loss[mod] = loss
            mod_n_examples[mod] = n_examples
            mod_preds[mod] = preds
            mod_targets[mod] = targets
                
        loss = sum(mod_loss.values()) if mod_loss else None

        return loss, mod_loss, mod_n_examples, mod_preds, mod_targets, static_targets, static_preds


    def forward_unimodal_output(
        self, mod_dict: Dict[str, Dict[str, torch.Tensor]], y
    ) -> MultiModalOutput:
        # Trick to handle incompatibility between unimodal and multimodal outputs
        # Can we improve this in the future?
        output_mod_dict = {}
        input_data = next(mod_dict[mod] for mod in self.encoder_embeddings if mod in mod_dict)
        eid = input_data["eid"]

        if self.model_mode == "encoding":
            mod_list = ["spike"]
        else:
            mod_list = self.avail_beh

        B, N, P = y.size()

        for mod in mod_list:

            output_mod_dict[mod] = {}
            if hasattr(self, "mod_stitcher_proj_dict"):
                y_mod = y.clone()
                if hasattr(self, "mod_static_weight_dict") and (mod in STATIC_VARS):
                    weight = torch.zeros_like(y, device=y.device) 
                    eid = np.array(eid)
                    unique_eids = np.unique(eid)
                    for group_eid in unique_eids:
                        mask = torch.tensor(np.argwhere(eid==group_eid), device=y.device).squeeze()
                        if mask.dim() > 0:
                            weight[mask] = self.mod_static_weight_dict[mod][group_eid][None,:,None].expand(mask.size(0),N,P)
                    y_mod = torch.sum(y.reshape(B,N,P) * weight, 1).reshape(B,-1)

                if self.model_mode == "encoding":
                    y_mod = y_mod
                preds = self.mod_stitcher_proj_dict[mod](y_mod, eid) 
                output_mod_dict[mod]["preds"] = preds.reshape((B,N,-1)) \
                    if mod not in STATIC_VARS else preds

            target_data = mod_dict.get(mod, input_data)
            output_mod_dict[mod]["temporal_mask"] = target_data["temporal_mask"]
            if mod == "spike":
                output_mod_dict[mod]["neuron_mask"] = self._neuron_mask(
                    eid, B, y.device, target_data.get("neuron_mask"))
            if "targets" in target_data and mod in mod_dict:
                output_mod_dict[mod]["targets_mask"] = target_data["targets_mask"]
                output_mod_dict[mod]["gt"] = target_data["targets"]
        
        return output_mod_dict


    def _prepare_mixed_masking(self, mod_dict):
                    
        tmp = mod_dict["spike"]["inputs"].clone()
        
        masking_schemes = [
            "encoding",
            "decoding",
            "self-spike",
            "self-vision",
            "random_token"
        ]
        selected_schemes = np.random.choice(
            masking_schemes, size=tmp.size()[0], replace=True
        )
        all_ones = torch.ones_like(tmp).to(tmp.device, torch.int64)
        all_zeros = all_ones * 0.
        
        mask_map = {}
        for mod in self.avail_mod:
            if mod == "spike":
                mask_map[mod] = {
                    "encoding": all_ones,
                    "decoding": all_zeros,
                    "self-spike": self.masker(tmp, None, "temporal")[1],
                    "self-vision": all_zeros,
                    "random_token": self.masker(tmp, None, "temporal")[1],
                }
            elif mod in DYNAMIC_VARS+STATIC_VARS:
                mask_map[mod] = {
                    "encoding": 1 - mask_map["spike"]["encoding"],
                    "decoding": 1 - mask_map["spike"]["decoding"],
                    "self-spike": all_zeros,
                    "self-vision": self.masker(tmp, None, "temporal")[1],
                    "random_token": mask_map["spike"]["random_token"],
                }
        return mask_map, selected_schemes

    
    def forward(self, mod_dict=None, **encoding_inputs) -> MultiModalOutput:
        if encoding_inputs:
            if mod_dict is not None:
                raise ValueError("Use either modality dictionaries or encoding keyword inputs")
            return self.encode(**encoding_inputs)
        if not isinstance(mod_dict, dict) or not mod_dict:
            raise ValueError("Provide modality dictionaries or visual encoding inputs")
        mod_dict = {mod: dict(values) for mod, values in mod_dict.items()}
        mixed = (self.training and self.model_mode == "mm"
                 and mod_dict.get("spike", {}).get("training_mode") == "mixed")
        if mixed and not self.mask:
            raise ValueError("Mixed stochastic masking requires an enabled training masker")
        if mixed:
            mask_map, selected_schemes = self._prepare_mixed_masking(mod_dict)

        for mod, d in mod_dict.items():
            if mod not in self.mod_type:
                raise ValueError(f"Unsupported modality {mod!r}")

            for name in ["inputs", "targets"]:
                if name in d and not isinstance(d[name], torch.Tensor):
                    raise ValueError(f"{mod} {name} must be a tensor")
                if name in d and len(mod_dict[mod][name].size()) == 2:
                    mod_dict[mod][name] = mod_dict[mod][name].unsqueeze(-1)
            values = d.get("inputs", d.get("targets"))
            if values is None or values.ndim != 3:
                raise ValueError(f"{mod} requires [B,T,D] inputs or targets")
            B, T, _ = values.shape
            width = self.visual_dim if mod == "vision-clip" else max(self.eid_list.values())
            if B == 0 or not 0 < T <= self.max_F or values.size(-1) != width:
                raise ValueError(f"Invalid {mod} batch, sequence length or feature width")
            for name in ("inputs", "targets"):
                tensor = d.get(name)
                if tensor is not None and (not isinstance(tensor, torch.Tensor)
                                          or tensor.shape != values.shape or tensor.device != values.device):
                    raise ValueError(f"{mod} inputs and targets must agree in shape and device")
            valid = self._valid_mask(d.get("temporal_mask", d.get("inputs_attn_mask")),
                                     (B, T), values.device, "temporal_mask")
            d["temporal_mask"] = valid
            if mod == "spike":
                d["neuron_mask"] = self._neuron_mask(d["eid"], B, values.device, d.get("neuron_mask"))
            if d.get("eval_mask") is not None:
                requested = d["eval_mask"]
                if (not isinstance(requested, torch.Tensor) or requested.ndim not in (2, 3)
                        or requested.shape[:2] != (B, T) or (requested.ndim == 3 and requested.size(-1) == 0)):
                    raise ValueError("eval_mask must have temporal dimensions [B,T]")
                self._valid_mask(requested, requested.shape, values.device, "eval_mask")
                mask = requested[..., 0] if requested.ndim == 3 else requested
                mask = self._valid_mask(mask, (B, T), values.device, "eval_mask")
                selection = mask & valid
            elif self.training and self.mask and d.get("training_mode") == "random_token":
                # Validity-safe input to the retained corruption sampler.
                safe_values = torch.where(valid.unsqueeze(-1), values, torch.zeros_like(values))
                _, requested = self.masker(safe_values.clone(), None)
                mask = requested[..., 0].bool()
                selection = mask & valid
            else:
                mask = torch.zeros_like(valid)
                selection = valid
            if mixed:
                mask_list = []
                for sample_idx, scheme in enumerate(selected_schemes):
                    tmp = mask_map[mod][scheme][sample_idx, :, 0].bool() & valid[sample_idx]
                    mask_list.append(tmp.unsqueeze(0))
                mask = torch.cat(mask_list, dim=0)
                selection = mask
            requested_corruption = d.get("training_mask", d.get("inputs_token_mask", mask))
            if (not isinstance(requested_corruption, torch.Tensor)
                    or requested_corruption.ndim not in (2, 3)
                    or requested_corruption.shape[:2] != (B, T)
                    or (requested_corruption.ndim == 3 and requested_corruption.size(-1) == 0)):
                raise ValueError("Training corruption must have temporal dimensions [B,T]")
            self._valid_mask(requested_corruption, requested_corruption.shape, values.device, "training_mask")
            if requested_corruption.ndim == 3:
                requested_corruption = requested_corruption[..., 0]
            corruption = self._valid_mask(requested_corruption, (B, T), values.device, "training_mask") & valid
            d["inputs_mask"] = corruption
            d["targets_mask"] = d.get("target_mask", selection)
            if "inputs" in d:
                positions = d.get("inputs_timestamp")
                if (not isinstance(positions, torch.Tensor) or positions.shape != (B, T)
                        or positions.dtype != torch.long or positions.device != values.device):
                    raise ValueError("Input positions must be int64 [B,T] on the input device")
                input_valid = (valid & ~corruption).unsqueeze(-1)
                if mod == "spike":
                    input_valid = input_valid & d["neuron_mask"].unsqueeze(1)
                d["inputs"] = torch.where(input_valid, d["inputs"], torch.zeros_like(d["inputs"]))
                d["inputs_timestamp"] = torch.where(valid, d["inputs_timestamp"],
                                                     torch.zeros_like(d["inputs_timestamp"]))

        encoder_inputs = [mod_dict[mod] for mod in self.encoder_embeddings if mod in mod_dict]
        if not encoder_inputs or any("inputs" not in d for d in encoder_inputs):
            raise ValueError("Missing required encoder input modality")
        if len(encoder_inputs) != len(self.encoder_embeddings):
            raise ValueError("Missing required encoder input modality")
        sessions = list(encoder_inputs[0]["eid"])
        if any(d["temporal_mask"].shape != encoder_inputs[0]["temporal_mask"].shape
               for d in mod_dict.values()):
            raise ValueError("Modality batch and temporal dimensions must agree")
        if any(list(d["eid"]) != sessions for d in mod_dict.values()):
            raise ValueError("Modality session identities must agree sample by sample")
        if self.model_mode in ("encoding", "decoding") and any(
                not torch.equal(d["temporal_mask"], encoder_inputs[0]["temporal_mask"])
                for d in mod_dict.values()):
            raise ValueError("Unimodal input and target temporal validity must agree")

        encoder_mod_dict = {
            mod: self.encoder_embeddings[mod](d)
            for mod, d in mod_dict.items() if mod in self.encoder_embeddings
        }
        encoder_tokens, encoder_emb, input_timestamp, encoder_mask, encoder_mod_mask = \
        self.forward_mask_encoder(encoder_mod_dict)

        x = encoder_tokens + encoder_emb
        validity = torch.cat([mod_dict[mod]["temporal_mask"] for mod in encoder_mod_dict], dim=1)
        x = self.forward_encoder(x, input_timestamp=input_timestamp, temporal_mask=validity)
        if "vision-clip" in self.avail_beh:
            x = F.normalize(x, dim=-1)
        if self.model_mode == "mm":
            output_mod_dict = {
                mod: self.encoder_embeddings[mod].out_proj(
                    self.mod_to_indx[mod], d, x, encoder_mod_mask, len(self.avail_mod)
                )
                for mod, d in encoder_mod_dict.items() if mod in self.encoder_embeddings
            }
        else:
            output_mod_dict = self.forward_unimodal_output(mod_dict, x)

        for mod, d in output_mod_dict.items():
            temporal = d["temporal_mask"]
            valid_outputs = temporal.unsqueeze(-1)
            if mod == "spike":
                d["neuron_mask"] = self._neuron_mask(sessions, temporal.size(0), x.device,
                                                      d.get("neuron_mask"))
                valid_outputs = valid_outputs & d["neuron_mask"].unsqueeze(1)
            d["preds"] = torch.where(valid_outputs, d["preds"], torch.zeros_like(d["preds"]))
            
        loss, mod_loss, mod_n_examples, mod_preds, mod_targets, static_targets, static_preds = \
        self.forward_loss(output_mod_dict)

        return MultiModalOutput(
            neural_prediction=mod_preds.get("spike"),
            temporal_mask=output_mod_dict.get("spike", next(iter(output_mod_dict.values())))["temporal_mask"],
            neuron_mask=output_mod_dict.get("spike", {}).get("neuron_mask"),
            loss=loss,
            mod_loss=mod_loss,
            mod_n_examples=mod_n_examples,
            mod_preds=mod_preds,
            mod_targets=mod_targets,
            static_preds=static_preds,
            static_targets=static_targets,
        )
