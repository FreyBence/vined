import numpy as np
import torch
from torch import nn

STATIC_VARS = []
DYNAMIC_VARS = ["vision-clip"]
OUTPUT_DIM = {
    "vision-clip": 768
}


def session_populations(eid_list):
    """Validate explicit session IDs and population widths without reordering."""
    if not isinstance(eid_list, dict) or not eid_list:
        raise ValueError("eid_list must be a nonempty session-to-neuron-count mapping")
    for session, count in eid_list.items():
        if not isinstance(session, str) or not session or "." in session:
            raise ValueError("Session IDs must be nonempty strings without dots")
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError(f"Invalid neuron count for session {session!r}")
    return {session: int(count) for session, count in eid_list.items()}


def session_indices(eid, populations, batch_size, device):
    """Return one-dimensional dispatch indices, including singleton groups."""
    if (not isinstance(eid, (list, tuple, np.ndarray))
            or (isinstance(eid, np.ndarray) and eid.ndim != 1) or len(eid) != batch_size):
        raise ValueError("Provide one explicit session ID per sample")
    groups = {}
    for index, session in enumerate(eid):
        if not isinstance(session, str) or session not in populations:
            raise ValueError(f"Unknown session {session!r}")
        groups.setdefault(session, []).append(index)
    return [(session, torch.tensor(indices, device=device, dtype=torch.long))
            for session, indices in groups.items()]


class StitchEncoder(nn.Module):
    def __init__(self, 
         eid_list: dict,
         n_channels: int,
         scale: int=1,
         mod: str="spike",
         max_F: int=100,
         visual_dim: int=768,
    ):
        super().__init__()

        self.mod = mod
        self.P = n_channels
        self.max_F = max_F
        self.visual_dim = visual_dim
        self.eid_list = session_populations(eid_list)
        self.N = max(self.eid_list.values())
        stitcher_dict, project_dict = {}, {}
        for key, val in self.eid_list.items():
            if mod == "vision-clip":
                val = visual_dim
            elif mod in STATIC_VARS:
                val = 1
            else:
                val = self.N
            mult = max_F if mod in STATIC_VARS else 1
            # projection layer
            if mod == "vision-clip":
                project_dict[str(key)] = nn.Sequential(
                    nn.LayerNorm(visual_dim),
                    nn.Linear(visual_dim, n_channels)
                )
            else:
                stitcher_dict[str(key)] = nn.Linear(
                    int(val),
                    int(val) * 2 * mult
                )
                project_dict[str(key)] = nn.Linear(
                    int(val) * 2,
                    n_channels
                )

        self.stitcher_dict = nn.ModuleDict(stitcher_dict)
        self.project_dict = nn.ModuleDict(project_dict)
        self.scale = scale
        self.act = nn.Softsign()

    def forward(self, x, eid):
        expected_width = self.visual_dim if self.mod == "vision-clip" else self.N
        if x.ndim != 3 or x.size(-1) != expected_width or not 0 < x.size(1) <= self.max_F:
            raise ValueError("Invalid stitch encoder feature width or sequence length")
        groups = session_indices(eid, self.eid_list, len(x), x.device)
        out = x.new_zeros((len(x), x.size(1), self.P))
        for group_eid, mask in groups:
            x_group = x[mask]
            if self.mod == "vision-clip":
                stitched = x_group
            else:
                stitched = self.stitcher_dict[group_eid](x_group)
                
            if self.mod in STATIC_VARS:
                stitched = stitched.reshape(stitched.shape[0], -1, 2)
            if self.mod == "vision-clip":
                stitched = stitched
            else:
                stitched = self.act(stitched) * self.scale
            out[mask] = self.project_dict[group_eid](stitched)
        return out


class StitchDecoder(nn.Module):
    def __init__(self,
         eid_list: dict,
         n_channels: int,
         mod:str="spike",
         max_F: int=100,
         visual_dim: int=768,
    ):
        super().__init__()
        
        self.mod = mod
        self.max_F = max_F
        self.P = n_channels
        self.eid_list = session_populations(eid_list)
        max_num_neuron = max(self.eid_list.values())
        stitch_decoder_dict = {}
        for key, val in self.eid_list.items():
            if mod in STATIC_VARS:
                val, mult = OUTPUT_DIM[mod], 1
            elif mod in DYNAMIC_VARS:
                val, mult = visual_dim, 1
            else:
                mult = 1

            if mod == "vision-clip":
                stitch_decoder_dict[str(key)] = nn.Sequential(
                    nn.Linear(n_channels, visual_dim),
                    nn.LayerNorm(visual_dim)
                )
            else:
                stitch_decoder_dict[str(key)] = nn.Sequential(
                    nn.Linear(n_channels * mult, val),
                    # Normalize only real channels; a singleton must retain its signal.
                    nn.LayerNorm(val) if val > 1 else nn.Identity()
                )

        self.stitch_decoder_dict = nn.ModuleDict(stitch_decoder_dict)
        self.N = max_num_neuron if mod == "spike" else val

    def forward(self, x, eid):
        if not isinstance(eid, (list, tuple, np.ndarray)) or not len(eid):
            raise ValueError("Provide a nonempty sequence of session IDs")
        groups = session_indices(eid, self.eid_list, len(eid), x.device)
        if (x.ndim not in (2, 3) or x.size(-1) != self.P
                or (x.ndim == 3 and x.size(0) != len(eid))
                or (x.ndim == 2 and x.size(0) % len(eid))):
            raise ValueError("Decoder latent width or batch/session dimensions disagree")
        x = x.reshape((len(eid), -1, self.P))
        B, T, _ = x.size()
        if not 0 < T <= self.max_F:
            raise ValueError("Decoder sequence length exceeds its configured maximum")
        out = x.new_zeros((B,T,self.N))
        for group_eid, mask in groups:
            x_group = x[mask]
            predictions = self.stitch_decoder_dict[group_eid](x_group)
            out[mask, :, :predictions.size(-1)] = predictions
        return out

    def neuron_mask(self, eid, *, device=None):
        """Real output channels in session-column order; padding is false."""
        if self.mod != "spike":
            raise ValueError("neuron_mask is only defined for neural output heads")
        if not isinstance(eid, (list, tuple, np.ndarray)):
            raise ValueError("Provide a sequence of session IDs")
        session_indices(eid, self.eid_list, len(eid), device)
        widths = torch.tensor([self.eid_list[session] for session in eid], device=device)
        return torch.arange(self.N, device=device)[None, :] < widths[:, None]
    
    

