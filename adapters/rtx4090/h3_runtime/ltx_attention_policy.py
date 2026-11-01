"""Editable RSI extension: LTX video self-attention only.

The protected runtime constructs this policy before building the transformer.
The policy never receives a model or audio/cross-attention module. Implementations
may use local research_sparse_attention helpers, but may not change the frozen
scheduler, prefix protection, attention evidence collector or evaluator.
"""
from __future__ import annotations


class DenseVideoAttentionPolicy:
    backend_id = 'dense'

    def attention(self, dense, *args, layer_index, step_index,
                  full_attention_prefix_tokens, **kwargs):
        """Identity control: execute the unchanged upstream dense callable."""
        return dense(*args, **kwargs)


def build_video_attention_policy(*, sdpa_backend):
    if sdpa_backend not in {'flash', 'efficient'}:
        raise ValueError('Unsupported frozen LTX SDPA backend')
    return DenseVideoAttentionPolicy()
