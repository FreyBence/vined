import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

ACT2FN["softsign"] = nn.Softsign
from models.stitcher import StitchDecoder, StitchEncoder, session_indices, session_populations
from multi_modal.mm_utils import MLP, Attention, ScaleNorm
from utils.config_utils import DictConfig, update_config

DEFAULT_CONFIG = "src/configs/multi_modal/mm.yaml"

STATIC_VARS = []
VISION_VARS = ["vision-clip"]


class EncoderEmbeddingLayer(nn.Module):
    def __init__(
        self, hidden_size, n_channels, config: DictConfig, stitching=False, eid_list=None, mod=None, max_F=100
    ):
        super().__init__()

        self.bias = config.bias
        self.n_channels = config.get("visual_dim", 768) if mod in VISION_VARS else n_channels
        self.input_dim = self.n_channels*config.mult
        self.max_F = max_F

        self.mod_emb = nn.Embedding(config.n_modality, hidden_size)

        self.eid_list = session_populations(eid_list)
        self.eid_lookup = list(self.eid_list)
        self.eid_to_indx = {r: i for i, r in enumerate(self.eid_lookup)}
        self.session_emb = nn.Embedding(len(self.eid_lookup), hidden_size)

        self.pos = config.pos
        if self.pos:
            self.pos_embed = nn.Embedding(max_F, hidden_size)

        self.dropout = nn.Dropout(config.dropout)

        if stitching:

            self.mod_stitch_encoder = StitchEncoder(
                eid_list=eid_list,
                n_channels=hidden_size,
                mod=mod,
                max_F=max_F,
                visual_dim=config.get("visual_dim", 768),
            )

        elif mod in VISION_VARS:

            self.vision_proj = nn.Sequential(
                nn.LayerNorm(self.n_channels),
                nn.Linear(self.n_channels, hidden_size)
            )

        else:

            self.token_embed = nn.Linear(
                self.n_channels,
                self.input_dim,
                bias=self.bias
            )

            self.projection = nn.Linear(
                self.input_dim,
                hidden_size
            )

            self.act = (
                ACT2FN[config.act]
                if config.act != "identity"
                else nn.Identity()
            )

            self.scale = (
                hidden_size ** 0.5
                if config.scale == None
                else config.scale
            )

    
    def forward(self, d: Dict[str, torch.Tensor]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:  

        inputs, inputs_timestamp, inputs_modality, eid = \
        d["inputs"], d["inputs_timestamp"], d["inputs_modality"], d["eid"]
        if (not isinstance(inputs_modality, torch.Tensor) or inputs_modality.ndim != 0
                or inputs_modality.dtype != torch.long or inputs_modality.device != inputs.device
                or not 0 <= inputs_modality.item() < self.mod_emb.num_embeddings):
            raise ValueError("inputs_modality must be a valid scalar int64 index on the input device")
        B, N, D = inputs.size()
        if not 0 < N <= self.max_F:
            raise ValueError("Input sequence length exceeds configured max_F")
        if (not isinstance(inputs_timestamp, torch.Tensor)
                or inputs_timestamp.shape != (B, N) or inputs_timestamp.dtype != torch.long
                or inputs_timestamp.device != inputs.device
                or torch.any(inputs_timestamp < 0) or torch.any(inputs_timestamp >= self.max_F)):
            raise ValueError("Positions must be int64 [B,T] within configured max_F")
        session_indices(eid, self.eid_list, B, inputs.device)
        if hasattr(self, "mod_stitch_encoder"):

            x = self.mod_stitch_encoder(inputs, eid)

        elif hasattr(self, "vision_proj"):
            inputs = F.normalize(inputs, dim=-1)
            x = self.vision_proj(inputs)

        else:

            x = self.token_embed(inputs)

            x = self.act(x) * self.scale

            x = self.projection(x)

        x_embed = self.mod_emb(inputs_modality)[None,None,:].expand(B,N,-1).clone()

        if self.pos:
            x_embed += self.pos_embed(inputs_timestamp)

        if (len(self.eid_lookup) != self.session_emb.num_embeddings
                or set(self.eid_lookup) != set(self.eid_list)
                or self.eid_to_indx != {session: index for index, session in enumerate(self.eid_lookup)}):
            raise ValueError("Session embedding identities do not match configured sessions")
        session_idx = torch.tensor([self.eid_to_indx[session] for session in eid],
                                   device=x.device, dtype=torch.long)
        x_embed += self.session_emb(session_idx)[:, None, :]

        return self.dropout(x), x_embed


class EncoderEmbedding(nn.Module):
    def __init__(
        self, 
        n_channel,
        output_channel,
        config: DictConfig,
        stitching=False,
        eid_list=None,
        mod=None,
        max_F=100,
        **kwargs
    ):
        super().__init__() 

        self.hidden_size = config.transformer.hidden_size
        self.n_layers = config.transformer.n_layers
        self.max_F = max_F
        self.n_channel = n_channel
        self.output_channel = output_channel
        self.visual_dim = config.embedder.get("visual_dim", 768)
        if (isinstance(self.visual_dim, bool) or not isinstance(self.visual_dim, int)
                or self.visual_dim <= 0 or isinstance(max_F, bool)
                or not isinstance(max_F, int) or max_F <= 0):
            raise ValueError("visual_dim and max_F must be positive integers")
        if mod in VISION_VARS:
            self.output_channel = self.visual_dim

        self.embedder = EncoderEmbeddingLayer(
            self.hidden_size, self.n_channel, config.embedder, stitching, eid_list, mod, max_F
        )

        if stitching:

            self.mod_stitcher_proj_dict = StitchDecoder(
                eid_list=eid_list,
                n_channels=self.hidden_size,
                mod=mod,
                max_F=max_F,
                visual_dim=self.visual_dim,
            )

            if mod in STATIC_VARS:
                ...
                
        elif mod in VISION_VARS:

            self.out = nn.Sequential(
                nn.Linear(self.hidden_size, self.output_channel),
                nn.LayerNorm(self.output_channel)
            )

        else:

            self.out = nn.Linear(
                self.hidden_size,
                self.output_channel
            )

    def forward(self, d : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:    
                        
        x, x_emb = self.embedder(d)
        d["x"], d["emb"] = x, x_emb
        if "targets" in d:
            d["gt"] = d["targets"]
        
        return d

    def out_proj(self, 
        mod_idx: int, d: Dict[str, torch.Tensor], y: torch.Tensor, 
        mod_mask: torch.Tensor, n_mod: int,
    ) -> Dict[str, torch.Tensor]: 

        B, N, P = y.size()

        y_mod = y[mod_mask == mod_idx]
        
        if hasattr(self, "mod_stitcher_proj_dict"):
            if hasattr(self, "mod_static_weight_dict"):
                weight = torch.zeros_like(y_mod.reshape(B,-1,P), device=y.device) 
                eid = np.array(d["eid"])
                unique_eids = np.unique(eid)
                for group_eid in unique_eids:
                    mask = torch.tensor(np.argwhere(eid==group_eid), device=y.device).squeeze()
                    if mask.dim() > 0:
                        weight[mask] = self.mod_static_weight_dict[group_eid][None,:,None].expand(mask.size(0),-1,P)
                y_mod = torch.sum(
                    y_mod.reshape(B,-1,P) * weight, 1
                ).reshape(B,-1)
            preds = self.mod_stitcher_proj_dict(y_mod, d["eid"])
            d["preds"] = preds.reshape((B,-1,preds.size()[-1])) \
                if not hasattr(self, "mod_static_weight_dict") else preds
        else:
            y_mod = self.out(y_mod).reshape((B,-1,self.output_channel))
            if self.output_channel > 1:
                d["preds"] = F.normalize(y_mod, dim=-1)
            else:
                d["preds"] = y_mod
        
        return d
        


class EncoderLayer(nn.Module):
    
    def __init__(self, idx, config: DictConfig, max_F=100):
        super().__init__()

        self.idx = idx
    
        self.ln1 = ScaleNorm(config.hidden_size ** 0.5) \
            if config.use_scalenorm else nn.LayerNorm(config.hidden_size) 
        self.attn = Attention(
            idx, config.hidden_size, config.n_heads, config.attention_bias, 
            config.dropout, config.use_rope, 
            max_F=max_F, n_mod=1,
        )
        self.ln2 = ScaleNorm(config.hidden_size ** 0.5) \
            if config.use_scalenorm else nn.LayerNorm(config.hidden_size) 
        self.mlp = MLP(
            config.hidden_size, config.inter_size, config.act, 
            config.mlp_bias, config.dropout
        )
        if config.fixup_init:
            self.fixup_initialization(config.n_layers)

    def forward(
        self, x: torch.FloatTensor, 
        mask: Optional[torch.LongTensor] = None, 
        timestamp: Optional[torch.LongTensor] = None,  
    ) -> torch.FloatTensor :                           
        
        x = x + self.attn(self.ln1(x), mask=mask, timestamp=timestamp)

        x = x + self.mlp(self.ln2(x))

        return x

    def fixup_initialization(self, n_layers):
        temp_state_dic = {}
        for name, param in self.named_parameters():
            if name.endswith("_proj.weight"):
                temp_state_dic[name] = (0.67 * (n_layers) ** (- 1./4.)) * param
            elif name.endswith("value.weight"):
                temp_state_dic[name] = (0.67 * (n_layers) ** (- 1./4.)) * (param * (2**0.5))
                
        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)
