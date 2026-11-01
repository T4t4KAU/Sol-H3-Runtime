"""Optional global GEMM shapes for numerical comparison of LTX sequence shards.

This spends extra compute to retain the unsharded Linear dispatch and BF16
rounding. Only temporary Linear inputs are expanded; attention never receives
padding tokens. Checkpoint tensors and native Linear arithmetic are unchanged.
"""
from dataclasses import dataclass

import torch


@dataclass
class LinearSequenceLayout:
    rank: int
    world: int
    global_tokens: int = 0
    local_tokens: int = 0

    def prepare(self, module, args, kwargs):
        video = kwargs.get('video', args[0] if args else None)
        if video is None:
            self.global_tokens = self.local_tokens = 0
            return
        count = video.latent.shape[-2]
        if count % self.world:
            raise ValueError('Global Linear shapes require equal video sequence partitions')
        self.global_tokens, self.local_tokens = count, count // self.world
        audio = kwargs.get('audio', args[1] if len(args) > 1 else None)
        # The native model uses the token dimension for video, text and audio.
        # Reject ambiguous lengths instead of applying a video layout to them.
        others = [video.context]
        if audio is not None:
            others.extend([audio.latent, audio.context])
        if self.world > 1 and any(x is not None and x.shape[-2] == self.local_tokens for x in others):
            raise ValueError('Video shard length collides with an unsharded modality/context length')

    def install(self, model):
        for layer in model.modules():
            if type(layer) is torch.nn.Linear:
                layer.__class__ = _GlobalShapeLinear
                layer._sequence_linear_layout = self


class _GlobalShapeLinear(torch.nn.Linear):
    def forward(self, value):
        layout = self._sequence_linear_layout
        if (layout.world == 1 or not layout.local_tokens or value.ndim < 2
                or value.shape[-2] != layout.local_tokens):
            return super().forward(value)
        shape = list(value.shape)
        shape[-2] = layout.global_tokens
        if value.stride(-1) == 1:
            full = value.new_zeros(shape)
        elif value.stride(-2) == 1:
            # H3->LTX video latents are feature-major. Making them contiguous
            # would switch native Linear from matmul+add to flattened addmm.
            full = value.new_zeros((*shape[:-2], shape[-1], shape[-2])).transpose(-1, -2)
        else:
            raise ValueError('Unvalidated Linear input strides for global sequence layout')
        start = layout.rank * layout.local_tokens
        full[..., start:start + layout.local_tokens, :] = value
        result = super().forward(full)
        # Release the global temporary before the next module; do not retain a view.
        return result[..., start:start + layout.local_tokens, :].clone()
