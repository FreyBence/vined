"""Fixed-weight CLIP vision attention using PyTorch's SDPA implementation."""

from torch import nn
from torch.nn import functional as F


class _VisionSDPAAttention(nn.Module):
    """Reuse the original projections; retain its path for masks/attention output."""

    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None,
                output_attentions=False):
        original = self.original
        if attention_mask is not None or causal_attention_mask is not None or output_attentions:
            return original(hidden_states, attention_mask=attention_mask,
                            causal_attention_mask=causal_attention_mask,
                            output_attentions=output_attentions)
        batch, length, width = hidden_states.shape

        def heads(projection):
            return projection(hidden_states).view(
                batch, length, original.num_heads, original.head_dim).transpose(1, 2)

        # SDPA applies head_dim**-0.5 itself; do not also scale the query.
        output = F.scaled_dot_product_attention(
            heads(original.q_proj), heads(original.k_proj), heads(original.v_proj),
            dropout_p=original.dropout if original.training else 0.0)
        output = output.transpose(1, 2).reshape(batch, length, width)
        return original.out_proj(output), None


def use_vision_sdpa(model):
    """Install only on the supported Hugging Face CLIP vision attention class."""
    from transformers.models.clip.modeling_clip import CLIPAttention

    layers = model.vision_model.encoder.layers
    if not layers or any(type(layer.self_attn) is not CLIPAttention for layer in layers):
        raise ValueError("SDPA requires standard CLIP vision attention layers")
    for layer in layers:
        layer.self_attn = _VisionSDPAAttention(layer.self_attn)
