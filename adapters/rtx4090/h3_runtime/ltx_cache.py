"""Build the released fixed prompt cache with sequential INT8 model residency."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from .upstream import ltx_paths, runtime


def build_cache(paths, output):
    ltx_paths(paths['ltx_root'])
    official = runtime(paths['sana_root'], 'cache_builder')
    cache_api = runtime(paths['sana_root'], 'prompt_cache')
    output = Path(output)
    if output.exists():
        return cache_api.load_cache(output, prompt=cache_api.FIXED_PROMPT, torch_module=torch)
    gemma_path, connector_path = Path(paths['offline_gemma']), Path(paths['offline_connector'])
    for name, path in [('text_encoder', gemma_path), ('connector', connector_path)]:
        if path.name != cache_api.RECIPE[name] or path.stat().st_size != cache_api.RECIPE[name+'_bytes']:
            raise ValueError('Unexpected offline checkpoint: '+str(path))
    device = torch.device('cuda', torch.cuda.current_device())
    print('cache: loading native INT8 Gemma', flush=True)
    builder, unused = official._text_builders(gemma_path, connector_path)
    del unused
    gemma = builder.build(device=device, dtype=None).eval().requires_grad_(False)
    gemma_audit = official._validate_model(gemma, 328, torch)
    stats = official.prompt_stats(gemma.tokenizer, cache_api.FIXED_PROMPT)
    with torch.inference_mode():
        outputs = gemma.encode([cache_api.FIXED_PROMPT])
        if len(outputs) != 1:
            raise RuntimeError('Expected exactly one fixed prompt')
        hidden, mask = outputs[0]
        hidden = tuple(t.detach().cpu() for t in hidden)
        mask = mask.detach().cpu()
    gemma.dispose()
    gemma.to_empty(device='meta')
    del gemma, outputs, builder
    gc.collect()
    torch.cuda.empty_cache()
    print('cache: Gemma released; loading native INT8 connector', flush=True)
    unused, builder = official._text_builders(gemma_path, connector_path)
    del unused
    connector = builder.build(device=device, dtype=None).eval().requires_grad_(False)
    connector.feature_extractor.to(dtype=torch.bfloat16)
    connector_audit = official._validate_model(connector, 96, torch)
    with torch.inference_mode():
        processed = connector.process_hidden_states(tuple(t.to(device) for t in hidden), mask.to(device))
        contexts = {name: getattr(processed, name+'_encoding').detach().cpu().contiguous()
                    for name in ('video', 'audio')}
    payload = dict(schema_version=1, prompt=cache_api.FIXED_PROMPT, recipe=dict(cache_api.RECIPE),
                   fingerprint=cache_api.fingerprint(cache_api.FIXED_PROMPT), contexts=contexts,
                   prompt_stats=stats,
                   tensor_manifest={n:dict(shape=list(t.shape), dtype=str(t.dtype)) for n,t in contexts.items()},
                   generation=dict(loading_strategy='sequential resident INT8 Gemma then connector',
                                   gemma=gemma_audit, connector=connector_audit,
                                   gemma_encode_calls=1, connector_process_calls=1,
                                   video_model_load_count=0, mask_reapplied_to_contexts=False,
                                   capability=list(torch.cuda.get_device_capability(device))))
    cache_api.validate_payload(payload, prompt=cache_api.FIXED_PROMPT, torch_module=torch)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial')
    torch.save(payload, temporary)
    temporary.replace(output)
    loaded = cache_api.load_cache(output, prompt=cache_api.FIXED_PROMPT, torch_module=torch)
    if any(not torch.equal(loaded.payload['contexts'][n], t) for n,t in contexts.items()):
        raise RuntimeError('Cache serialization changed native outputs')
    output.with_suffix('.json').write_text(json.dumps(dict(status='PASS',
        saved_contexts_bitwise_equal_fresh=True, **payload['generation']), indent=2))
    connector.dispose()
    connector.to_empty(device='meta')
    del connector, builder, processed, hidden
    gc.collect()
    torch.cuda.empty_cache()
    return loaded
