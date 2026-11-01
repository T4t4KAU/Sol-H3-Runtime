"""FSDP at Qwen's actual `.model(...)` entry used by the official H3 encoder."""
import gc
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, Qwen3VLForConditionalGeneration

from .distributed_loading import load_cpu_source


@torch.no_grad()
def load_text_encoder(engine, path):
    source = None
    if engine.rank == 0:
        source = Qwen3VLForConditionalGeneration.from_pretrained(
            str(path), dtype=torch.bfloat16, local_files_only=True,
            attn_implementation='sdpa', low_cpu_mem_usage=True,
        ).eval().requires_grad_(False)
    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    with torch.device('meta'):
        target = Qwen3VLForConditionalGeneration._from_config(
            config, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    units=[]
    for _, module in target.named_modules():
        if type(module).__name__ in {'Qwen3VLTextDecoderLayer', 'Qwen3VLVisionBlock'} or isinstance(module, torch.nn.Embedding):
            units.append(module)
    # The head is unused but still loaded and sharded; do not leave meta tensors.
    units.extend([target.model.visual, target.model.language_model, target.model, target.lm_head])
    audit = engine.runtime.evidence_dir
    target = load_cpu_source(target, source, mesh=engine._get_parallel_mesh('fsdp'),
                            cpu_group=engine.cpu_group, units=units,
                            audit_dir=None if audit is None else Path(audit)/'qwen-weights')
    # Transformers 5.8 keys its output recorder by the exact class name. FSDP2
    # creates subclasses; retain the original recorder recipe for those classes.
    # The native hooks still collect all layers and Diffusers still selects [50].
    from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY
    for module in target.modules():
        cls = type(module)
        if str(cls) not in _CAN_RECORD_REGISTRY:
            for base in cls.__mro__[1:]:
                if str(base) in _CAN_RECORD_REGISTRY:
                    _CAN_RECORD_REGISTRY[str(cls)] = _CAN_RECORD_REGISTRY[str(base)]
                    break
    del source
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier(group=engine.cpu_group)
    return target
