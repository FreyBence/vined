"""Trial-local runtime indexing; scientific observations remain unchanged."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


TARGET_SUPPORT_BINS = 12
STRICT_CONTEXT_BINS = (1, 3, 6, 9, 12)


@dataclass(frozen=True)
class TemporalContextView:
    """Indices address each batch row's own temporal axis, never another trial."""

    direction: str
    mode: str
    context_bins: int | None
    context_indices: torch.Tensor | None
    context_valid: torch.Tensor
    target_mask: torch.Tensor
    readout_index: int | None
    target_support_bins: int = TARGET_SUPPORT_BINS


def temporal_context_view(temporal_positions, temporal_mask, *, direction,
                          mode, context_bins=None):
    """Return strict windows or full-trial context and separate fixed-H targets.

    Accept int64 positions and boolean validity shaped [T] or [B,T], as NumPy
    arrays or tensors. Results always have a batch axis and retain tensor device.
    Entire incomplete strict windows use -1; never gather those indices directly.
    Empty target selections are returned for caller-owned coverage/error handling.
    """
    import torch

    if direction not in ("encoding", "decoding"):
        raise ValueError("Context direction must be encoding or decoding")
    if mode not in ("strict", "full_trial"):
        raise ValueError("Context mode must be strict or full_trial")
    if mode == "strict":
        if type(context_bins) is not int or context_bins not in STRICT_CONTEXT_BINS:
            raise ValueError("Strict context requires bins in 1, 3, 6, 9, 12")
    elif context_bins is not None:
        raise ValueError("Full-trial context does not accept context_bins")
    positions = torch.as_tensor(temporal_positions)
    valid = torch.as_tensor(temporal_mask)
    if (positions.dtype != torch.int64 or valid.dtype != torch.bool
            or positions.shape != valid.shape or positions.ndim not in (1, 2)
            or positions.device != valid.device or positions.numel() == 0):
        raise ValueError("Positions/validity must be matching nonempty int64/bool [T] or [B,T] on one device")
    if positions.ndim == 1:
        positions, valid = positions.unsqueeze(0), valid.unsqueeze(0)
    if (positions[valid] < 0).any():
        raise ValueError("Real observations require nonnegative temporal positions")
    for row, mask in zip(positions, valid):
        real = row[mask]
        if (real[1:] <= real[:-1]).any():
            raise ValueError("Real temporal positions must be strictly increasing within each trial")
        if mode == "full_trial":
            real_indices = torch.nonzero(mask, as_tuple=True)[0]
            if ((real[1:] - real[:-1] != 1).any()
                    or (real_indices[1:] - real_indices[:-1] != 1).any()):
                raise ValueError("Full-trial context requires continuous real temporal support")

    b, t = positions.shape
    anchors = torch.arange(t, device=positions.device)
    safe_positions = torch.where(valid, positions, torch.zeros_like(positions))

    def windows(length):
        offsets = torch.arange(length, device=positions.device)
        if direction == "encoding":
            offsets = offsets - (length - 1)
        indices = anchors[:, None] + offsets
        in_bounds = ((indices >= 0) & (indices < t)).all(-1)
        safe = indices.clamp(0, t - 1)
        selected_valid = valid[:, safe]
        selected_positions = safe_positions[:, safe]
        # Differences avoid integer overflow when adding offsets to positions.
        consecutive = (selected_positions[:, :, 1:] - selected_positions[:, :, :-1] == 1).all(-1)
        complete = in_bounds[None, :] & selected_valid.all(-1) & consecutive
        indices = indices.expand(b, -1, -1).clone()
        return indices.masked_fill(~complete[:, :, None], -1), complete

    _, targets = windows(TARGET_SUPPORT_BINS)
    if mode == "full_trial":
        return TemporalContextView(direction, mode, None, None, valid.clone(),
                                   targets, None)
    indices, complete = windows(context_bins)
    return TemporalContextView(direction, mode, context_bins, indices, complete,
                               targets, context_bins - 1 if direction == "encoding" else 0)
