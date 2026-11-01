"""NCCL transport for the upstream LTX sequence-parallel attention wrappers."""
from __future__ import annotations

import torch
import torch.distributed as dist


class _AllToAll:
    def __init__(self, group):
        self.group = group
        self.world = dist.get_world_size(group)
        self.tokens = None

    def set_rank_tokens(self, counts):
        if len(counts) != self.world or len(set(counts)) != 1 or min(counts) < 1:
            raise ValueError("Only nonempty equal LTX video partitions are validated")
        self.tokens = tuple(counts)

    def send_recv_heads(self, value, copy_out=False):
        """[B,S/local,H,D] -> [B,S,H/local,D], preserving upstream layout."""
        batch, tokens, heads, dim = value.shape
        if self.tokens is None or tokens != self.tokens[0] or heads % self.world:
            raise ValueError("LTX all-to-all input does not match sequence/head partition")
        send = value.reshape(batch, tokens, self.world, heads // self.world, dim)
        send = send.permute(2, 0, 1, 3, 4).contiguous()
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send, group=self.group)
        return received.permute(1, 0, 2, 3, 4).reshape(
            batch, tokens * self.world, heads // self.world, dim).contiguous()

    def gather_heads(self, value, copy_out=False):
        """Inverse shuffle: [B,S,H/local,D] -> [B,S/local,H,D]."""
        batch, tokens, heads, dim = value.shape
        if self.tokens is None or tokens != sum(self.tokens):
            raise ValueError("LTX head gather does not match sequence partition")
        send = value.reshape(batch, self.world, tokens // self.world, heads, dim)
        send = send.permute(1, 0, 2, 3, 4).contiguous()
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send, group=self.group)
        return received.permute(1, 2, 0, 3, 4).reshape(
            batch, tokens // self.world, heads * self.world, dim).contiguous()


class NCCLAttentionManager:
    """Duck-typed AttentionManager; attention and modality logic remain upstream."""
    def __init__(self, group, max_tokens=65536):
        self.group, self.max_tokens = group, max_tokens
        self.rank, self.world_size = dist.get_rank(group), dist.get_world_size(group)
        self.copy_out = False
        self.all2all_q, self.all2all_k, self.all2all_v, self.all2all_heads = (
            _AllToAll(group) for _ in range(4))

    def set_seqlen_all2all(self, counts):
        for transport in (self.all2all_q, self.all2all_k, self.all2all_v, self.all2all_heads):
            transport.set_rank_tokens(counts)

    def send_recv_qkv(self, q, k, v):
        return (self.all2all_q.send_recv_heads(q), self.all2all_k.send_recv_heads(k),
                self.all2all_v.send_recv_heads(v))

    def gather_heads(self, value):
        return self.all2all_heads.gather_heads(value)


def require_divisible_video(module, args, kwargs):
    video = kwargs.get('video', args[0] if args else None)
    world = module.attention_manager.world_size
    if video is not None and video.latent.shape[1] % world:
        raise ValueError("LTX video tokens must divide SP degree; padded AV cross-attention is unvalidated")
