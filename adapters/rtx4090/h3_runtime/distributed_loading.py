"""Mixed-dtype FSDP2 loading from an independent rank-zero CPU model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


@torch.no_grad()
def load_cpu_source(target, source, *, mesh, cpu_group, units, audit_dir=None):
    """Call on every rank. `source` exists only on rank zero; target starts on meta."""
    rank = dist.get_rank()
    device = torch.device('cuda', torch.cuda.current_device())
    metadata = [None]
    if rank == 0:
        if any(p.device.type != 'cpu' for p in source.parameters()):
            raise ValueError('The independent source must remain on CPU')
        metadata[0] = {
            'parameters': [(n, tuple(p.shape), p.dtype) for n, p in source.named_parameters()],
            'buffers': [(n, tuple(b.shape), b.dtype) for n, b in source.named_buffers()],
        }
    print(f'rank={rank} FSDP source ready; synchronizing layout', flush=True)
    dist.broadcast_object_list(metadata, 0, group=cpu_group)
    metadata = metadata[0]
    actual = dict(target.named_parameters())
    if set(actual) != {n for n, _, _ in metadata['parameters']}:
        raise RuntimeError('Source/target parameter names differ')
    if set(dict(target.named_buffers())) != {n for n, _, _ in metadata['buffers']}:
        raise RuntimeError('Source/target buffer names differ')
    for name, shape, dtype in metadata['parameters']:
        p = actual[name]
        if tuple(p.shape) != shape:
            raise RuntimeError('Parameter shape mismatch: '+name)
        p.data = p.data.to(dtype=dtype)
    for name, shape, dtype in metadata['buffers']:
        parent, _, leaf = name.rpartition('.')
        target.get_submodule(parent)._buffers[leaf] = torch.empty(shape, dtype=dtype, device='meta')
    target.eval().requires_grad_(False)
    if mesh is not None:
        options = dict(mesh=mesh, reshard_after_forward=True,
                       mp_policy=MixedPrecisionPolicy(param_dtype=None, output_dtype=None,
                                                      cast_forward_inputs=False))
        seen = set()
        for unit in units:
            if id(unit) not in seen:
                fully_shard(unit, **options)
                seen.add(id(unit))
        if id(target) not in seen:
            fully_shard(target, **options)
    print(f'rank={rank} parameter layout installed; allocating CUDA storage', flush=True)
    target.to_empty(device=device)
    print(f'rank={rank} FSDP local allocation complete; broadcasting weights', flush=True)
    result = set_model_state_dict(target, dict(source.state_dict()) if rank == 0 else {},
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True,
                                 strict=True, ignore_frozen_params=False))
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f'Incomplete distributed load: {result}')
    print(f'rank={rank} FSDP weights loaded; checking buffers and shards', flush=True)
    # Includes nonpersistent rotary/frequency buffers not present in state_dict.
    for name, shape, dtype in metadata['buffers']:
        b = target.get_buffer(name)
        if rank == 0:
            b.copy_(source.get_buffer(name))
        dist.broadcast(b, 0)
        if b.is_meta or b.dtype != dtype or tuple(b.shape) != shape or not torch.isfinite(b).all():
            raise RuntimeError('Invalid materialized buffer: '+name)
    if [(n, tuple(p.shape), p.dtype) for n, p in target.named_parameters()] != metadata['parameters']:
        raise RuntimeError('Loaded parameter layout differs from the CPU source')
    audit_shards(target, source, cpu_group=cpu_group, directory=audit_dir,
                 sharded=mesh is not None)
    return target


def audit_shards(target, source, *, cpu_group, directory, sharded=True):
    """Compare one local shard at a time to the independent CPU source."""
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda', torch.cuda.current_device())
    rows = []
    for index, (name, p) in enumerate(target.named_parameters()):
        if p.requires_grad or (sharded and (not isinstance(p, DTensor) or p.placements != (Shard(0),))):
            raise RuntimeError('Unexpected parameter layout: '+name)
        if not sharded and isinstance(p, DTensor):
            raise RuntimeError('Replicated loading unexpectedly produced a shard: '+name)
        local = p.to_local() if sharded else p
        if local.is_meta or local.device != device:
            raise RuntimeError('Unmaterialized local shard: '+name)
        if directory is not None:
            hashes = [None]
            if rank == 0:
                full = source.get_parameter(name)
                if sharded:
                    stride = (full.shape[0]+world-1)//world
                    hashes[0] = [tensor_digest(full[min(r*stride,len(full)):min((r+1)*stride,len(full))])
                                 for r in range(world)]
                else:
                    hashes[0] = [tensor_digest(full)] * world
            dist.broadcast_object_list(hashes, 0, group=cpu_group)
            digest = tensor_digest(local)
            if digest != hashes[0][rank]:
                raise RuntimeError('Shard content differs from CPU source: '+name)
            rows.append(dict(name=name, shape=list(local.shape), dtype=str(local.dtype),
                             bytes=local.numel()*local.element_size(), sha256=digest))
            if rank == 0 and index % 128 == 0:
                print(f'FSDP shard contents verified: {index} {name}', flush=True)
    if directory is not None:
        path=Path(directory); path.mkdir(parents=True, exist_ok=True)
        (path/f'shards-rank{rank}.json').write_text(json.dumps(dict(
            status='PASS', parameters=rows, local_bytes=sum(r['bytes'] for r in rows),
            buffers={n:tensor_digest(b) for n,b in target.named_buffers()}), indent=2))
