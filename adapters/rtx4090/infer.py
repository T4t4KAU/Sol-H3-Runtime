#!/usr/bin/env python3
"""Single-stage FastH3, two-stage SOL-H3, or original H3 inference.

Launch policy belongs to run_*_8x4090.sh. This entry point also retains the
resident, unsharded and optional official kernel paths for other deployments.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def _forward_indices(value):
    try:
        indices = tuple(int(part.strip()) for part in value.split(','))
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError('Use comma-separated forward indices') from exc
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= 49 for i in indices):
        raise argparse.ArgumentTypeError('Forward indices must be unique values from 0 through 48')
    return tuple(sorted(indices))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('single-stage', 'two-stage', 'original'), default='single-stage',
                        help='FastH3 4 updates, H3 4 + LTX 3 updates, or H3 49 updates without LoRA')
    parser.add_argument('--model', type=Path, help='Local MiniMax-H3 checkpoint directory')
    parser.add_argument('--adapter', type=Path, help='FastH3 adapter; forbidden in original mode')
    parser.add_argument('--checkpoints', type=Path, help='Optional asset root; required for two-stage assets')
    parser.add_argument('--attention-backend', choices=('dense', 'sol', 'sol_bsa'), default='sol_bsa')
    parser.add_argument('--compute-quant', choices=('none', 'mxfp8'), default='none')
    parser.add_argument('--lora-mode', choices=('merged', 'separate', 'fused'), default='merged')
    parser.add_argument('--task', choices=('t2v', 'i2v', 'ref2va'), default='t2v')
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument('--prompt')
    prompts.add_argument('--prompt-file', type=Path)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--reference', action='append', default=[], metavar='TYPE:PATH',
                        help='Ordered Ref2VA input: image:, video:, or audio:')
    parser.add_argument('--reference-image-resize-mode', choices=('match', 'diffusers'), default='match')
    parser.add_argument('--duration', type=int, choices=(5, 10, 15), default=5)
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, help='Output MP4 file')
    parser.add_argument('--warmup', action='store_true', help='Run an extra identical request before generating')
    parser.add_argument('--fsdp-degree', type=int, default=None, help='1 for replicated weights, or world size')
    parser.add_argument('--ulysses-degree', '--sp-degree', type=int, default=None, help='1 or world size')
    parser.add_argument('--staged-loading', action='store_true')
    parser.add_argument('--flash-sdpa', action='store_true', help='Restrict H3 DiT SDPA to Flash')
    for flag, default in [('vae-batched', None), ('vae-global-batch', None), ('tf32', None),
                          ('bf16-reduced-precision-reduction', None), ('adaln-precompute', None),
                          ('kernel-fusions', True), ('packed-sp', True), ('text-fsdp', False),
                          ('parallel-vae', True)]:
        parser.add_argument('--'+flag, action=argparse.BooleanOptionalAction, default=default)
    parser.add_argument('--vae-compile-mode', choices=('eager', 'default', 'max-autotune-no-cudagraphs'))
    parser.add_argument('--vae-tile-size', type=int)
    parser.add_argument('--evidence-dir', type=Path,
                        help='Opt-in prediction/weight/resource capture; must be empty. Adds substantial overhead.')
    instrumentation = parser.add_mutually_exclusive_group()
    instrumentation.add_argument('--profile-dir', type=Path,
                        help='Original49 or two-stage: bounded CPU/CUDA traces, stage counts and latent fingerprints.')
    instrumentation.add_argument('--benchmark-dir', type=Path,
                        help='Original49 or two-stage: clean request timing, counts and allocator peaks; no profiler.')
    parser.add_argument('--raw-output-path', type=Path,
                        help='Benchmark or seven-step request/window profile: save raw outputs after timing; no overwrite.')
    parser.add_argument('--profile-forward-indices', type=_forward_indices, default=None,
                        help='H3 forward windows: original defaults 0,24,48; two-stage defaults 0,2,3; requires --profile-dir.')
    parser.add_argument('--profile-ltx-forward-indices', type=_forward_indices, default=None,
                        help='LTX forward windows, default 0,1,2; requires two-stage --profile-dir.')
    parser.add_argument('--profile-scope', choices=('forwards', 'request', 'windows'), default='forwards')
    parser.add_argument('--profile-window', action='append', default=[],
                        help='Exact seven_step/h3 or seven_step/ltx phase/forward region; repeatable.')
    parser.add_argument('--profile-record-shapes', action='store_true')
    parser.add_argument('--profile-with-stack', action='store_true')
    parser.add_argument('--profile-memory', action='store_true')
    parser.add_argument('--profile-collective-identifiers', action='store_true')
    parser.add_argument('--sana-root', type=Path, help='Pinned official Sana source checkout (two-stage only)')
    parser.add_argument('--ltx-root', type=Path, help='Pinned official LTX source checkout (two-stage only)')
    parser.add_argument('--upscaler-source', type=Path, help='Official H3 upscaler Python source (two-stage only)')
    parser.add_argument('--prompt-cache', type=Path, help='Validated fixed LTX prompt cache (two-stage only)')
    parser.add_argument('--ltx-attention-backend', choices=('flash', 'efficient'), default='flash')
    parser.add_argument('--ltx-linear-mode', choices=('native', 'full-sequence'), default='native')
    parser.add_argument('--resume-h3', type=Path, help='Reuse a matching H3 capture in two-stage mode')
    parser.add_argument('--build-prompt-cache', action='store_true',
                        help='Build the LTX cache once using its separate text-encoder environment; run without torchrun')
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser())
    if args.build_prompt_cache:
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            parser.error('Build the prompt cache in a single process, without the eight-GPU launcher')
        for name in ('checkpoints', 'sana_root', 'ltx_root', 'prompt_cache'):
            if getattr(args, name) is None:
                parser.error('--build-prompt-cache requires --'+name.replace('_', '-'))
        return args
    if args.model is None and args.checkpoints is not None:
        roots = [args.checkpoints/name for name in ('MiniMax-H3-Diffusers', 'MiniMax-H3')]
        found = [p for p in roots if p.is_dir()]
        if len(found) == 1:
            args.model = found[0]
    if args.mode != 'original' and args.adapter is None and args.checkpoints is not None:
        roots = [args.checkpoints/name/'dense-datafree/adapter_model.safetensors'
                 for name in ('FastH3', 'FastH3-4-step-Preview-v1-LoRA')]
        found = [p for p in roots if p.is_file()]
        if len(found) == 1:
            args.adapter = found[0]
    if args.model is None or not args.model.is_dir():
        parser.error('Provide an existing --model directory (or an unambiguous --checkpoints root)')
    if args.mode == 'original':
        if args.adapter is not None or args.lora_mode != 'merged':
            parser.error('Original H3 uses no adapter or LoRA branches')
        if args.adaln_precompute:
            parser.error('Original 49-step H3 requires --no-adaln-precompute')
        args.adaln_precompute = False
    elif args.adapter is None or not args.adapter.is_file():
        parser.error('FastH3 requires an existing --adapter file')
    if args.adaln_precompute is None:
        args.adaln_precompute = True
    if args.output is None:
        parser.error('--output is required')
    if args.prompt_file is not None:
        args.prompt = args.prompt_file.read_text(encoding='utf-8').strip()
    if not args.prompt or not args.prompt.strip():
        parser.error('Provide a nonempty --prompt or --prompt-file')
    if args.task == 'i2v' and args.image is None:
        parser.error('--image is required for i2v')
    if args.task != 'i2v' and args.image is not None:
        parser.error('--image is only valid for i2v; use --reference for Ref2VA')
    if args.task == 'ref2va' and not args.reference:
        parser.error('Ref2VA requires --reference')
    if args.task != 'ref2va' and args.reference:
        parser.error('--reference is only valid for Ref2VA')
    if args.task == 'ref2va' and args.attention_backend == 'sol':
        parser.error('Ref2VA supports dense or sol_bsa attention')
    for name in ('width', 'height'):
        value = getattr(args, name)
        if value is not None and (value <= 0 or value % 32):
            parser.error('--'+name+' must be a positive multiple of 32')
    if args.mode == 'two-stage':
        if args.task != 't2v' or args.duration != 5 or not args.staged_loading:
            parser.error('Two-stage generation requires staged T2V with duration 5')
        if args.width not in (None, 672) or args.height not in (None, 384):
            parser.error('Two-stage H3 geometry is fixed at 672x384; final LTX output is 1344x768')
        for name in ('checkpoints', 'sana_root', 'ltx_root', 'upscaler_source', 'prompt_cache'):
            if getattr(args, name) is None:
                parser.error('Two-stage generation requires --'+name.replace('_', '-'))
        if args.warmup:
            parser.error('--warmup is supported by the single-stage and original entry points')
    elif args.resume_h3 is not None:
        parser.error('--resume-h3 is only valid for two-stage generation')
    if args.evidence_dir is not None:
        if not args.staged_loading or args.warmup:
            parser.error('Evidence capture requires staged loading without --warmup')
        if args.evidence_dir.exists() and any(args.evidence_dir.iterdir()):
            parser.error('--evidence-dir must be empty to prevent mixing requests')
    artifact_dir = args.profile_dir or args.benchmark_dir
    if args.profile_scope != 'forwards':
        if args.mode != 'two-stage' or args.profile_dir is None:
            parser.error('Request/window profiling requires two-stage --profile-dir')
        if args.profile_forward_indices is not None or args.profile_ltx_forward_indices is not None:
            parser.error('Use --profile-window for request/window profiling, not legacy forward indices')
    if bool(args.profile_window) != (args.profile_scope == 'windows'):
        parser.error('--profile-window is required only with --profile-scope windows')
    if any((args.profile_record_shapes, args.profile_with_stack, args.profile_memory, args.profile_collective_identifiers)) and args.profile_scope == 'forwards':
        parser.error('Detailed profiling options require request or windows scope')
    if args.raw_output_path is not None:
        if args.benchmark_dir is None and not (args.mode == 'two-stage' and args.profile_scope != 'forwards'):
            parser.error('--raw-output-path requires --benchmark-dir')
        if args.raw_output_path.exists():
            parser.error('--raw-output-path must not already exist')
    if args.profile_forward_indices is not None and args.profile_dir is None:
        parser.error('--profile-forward-indices requires --profile-dir')
    if args.profile_ltx_forward_indices is not None and (args.profile_dir is None or args.mode != 'two-stage'):
        parser.error('--profile-ltx-forward-indices requires two-stage --profile-dir')
    if artifact_dir is not None:
        if args.mode not in ('original', 'two-stage') or not args.staged_loading or args.warmup:
            parser.error('Profiling/benchmark artifacts require original or two-stage mode and staged loading without --warmup')
        if args.evidence_dir is not None:
            parser.error('Profiling, benchmark timing and heavy evidence require separate runs')
        if args.mode == 'original' and os.environ.get('DISABLE_ADDMM_CUDA_LT', '0') != '0':
            parser.error('Original49 profiling/benchmark requires DISABLE_ADDMM_CUDA_LT=0')
        if args.mode == 'two-stage':
            if os.environ.get('DISABLE_ADDMM_CUDA_LT') != '1':
                parser.error('Seven-step profiling/benchmark requires DISABLE_ADDMM_CUDA_LT=1')
            if args.resume_h3 is not None:
                parser.error('Seven-step profiling/benchmark must execute the complete request; --resume-h3 is forbidden')
            if args.ltx_linear_mode != 'full-sequence':
                parser.error('Seven-step profiling/benchmark requires --ltx-linear-mode full-sequence')
            if args.profile_forward_indices is not None and any(i >= 4 for i in args.profile_forward_indices):
                parser.error('Seven-step H3 profiling indices must be from 0 through 3')
            if args.profile_ltx_forward_indices is not None and any(i >= 3 for i in args.profile_ltx_forward_indices):
                parser.error('Seven-step LTX profiling indices must be from 0 through 2')
        if artifact_dir.exists() and (not artifact_dir.is_dir() or any(artifact_dir.iterdir())):
            parser.error('Profiling/benchmark directory must be empty to prevent mixing requests')
    if args.profile_forward_indices is None:
        args.profile_forward_indices = (0, 2, 3) if args.mode == 'two-stage' else (0, 24, 48)
    if args.profile_ltx_forward_indices is None and args.mode == 'two-stage':
        args.profile_ltx_forward_indices = (0, 1, 2)
    if args.mode != 'two-stage':
        # Original49 runner receipts bind the exact argument object. A field
        # belonging only to the separate seven-step study must not change it.
        for field in ('profile_ltx_forward_indices', 'profile_scope', 'profile_window',
                      'profile_record_shapes', 'profile_with_stack', 'profile_memory', 'profile_collective_identifiers'):
            delattr(args, field)
    if getattr(args, 'profile_collective_identifiers', False):
        from h3_runtime.collective_profiling import configure_collective_profiling
        configure_collective_profiling()
    return args


def load_references(specs):
    if not specs:
        return None
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
    references, kinds = [], []
    for spec in specs:
        kind, separator, path = spec.partition(':')
        kind, path = kind.lower().strip(), path.strip()
        if not separator or kind not in {'image', 'video', 'audio'} or not path:
            raise ValueError(f'Invalid reference {spec!r}; use image:PATH, video:PATH, or audio:PATH')
        references.append(MiniMaxH3Reference(**{kind: path}))
        kinds.append(kind)
    if not ({'image', 'video'} & set(kinds)):
        raise ValueError('Ref2VA audio must be paired with an image or video reference')
    return references


def stage2_paths(args):
    root = args.checkpoints.expanduser().resolve()
    ltx = root/'LTX-2.5'
    paths = dict(sana_root=str(args.sana_root.resolve()), ltx_root=str(args.ltx_root.resolve()),
        offline_gemma=str(ltx/'text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors'),
        offline_connector=str(ltx/'diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors'),
        transformer=str(ltx/'diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors'),
        refiner_lora=str(ltx/'loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors'),
        audio_vae=str(ltx/'vae/ltx-2.5-audio-vae-bf16.safetensors'),
        output_video_vae=str(ltx/'vae/ltx-2.5-video-vae-conv-bf16.safetensors'),
        adapter_dir=str(root/'H3-to-LTX-Latent-Adapter'),
        h3_upscaler_source=None if args.upscaler_source is None else str(args.upscaler_source.resolve()),
        h3_upscaler_checkpoint=str(root/'H3-upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors'),
        prompt_cache=str(args.prompt_cache.resolve()))
    return paths


def run_two_stage(engine, args, paths):
    import torch
    from h3_runtime.two_stage import refine
    from h3_runtime.upstream import runtime
    cache_api = runtime(paths['sana_root'], 'prompt_cache')
    cache_api.load_cache(paths['prompt_cache'], prompt=cache_api.FIXED_PROMPT, torch_module=torch)
    payload = None
    if args.resume_h3 is None:
        media = engine.generate(args.prompt, duration=5, seed=args.seed,
                                width=672, height=384, capture_latents=True, num_inference_steps=5)
        if engine.is_rank_zero:
            audio = media.audio
            if (media.audio_sample_rate != 32000 or audio.dtype != torch.float32
                    or audio.ndim != 3 or tuple(audio.shape[:2]) != (1, 2) or audio.shape[2] < 161333
                    or not bool(torch.isfinite(audio).all())):
                raise RuntimeError('H3 PCM must be finite FP32 [1,2,N] at 32kHz')
            payload = dict(h3_normalized=media.video.to(torch.bfloat16).contiguous(),
                           audio=audio[:, :, :161333].clamp(-1, 1).contiguous())
            if args.evidence_dir is not None:
                torch.save(dict(native=media.video, payload=payload, prompt=args.prompt, seed=args.seed),
                           args.evidence_dir/'h3-capture.pt')
        del media
    elif engine.is_rank_zero:
        capture = torch.load(args.resume_h3, map_location='cpu', weights_only=True)
        if capture['prompt'] != args.prompt or capture['seed'] != args.seed:
            raise ValueError('Resume capture prompt/seed mismatch')
        payload = capture['payload']
    return refine(engine, paths, payload, args.output, seed=args.seed,
           evidence_dir=None if args.evidence_dir is None else args.evidence_dir/'ltx',
           attention_backend=args.ltx_attention_backend, linear_mode=args.ltx_linear_mode,
           collect_raw=engine.instrumentation is not None)


def run_instrumented_two_stage(engine, args, paths, recorder, torch_module):
    """Time all ranks through raw output packaging, then save and encode."""
    torch = torch_module
    if engine.world_size > 1:
        torch.distributed.barrier()
    torch.cuda.synchronize(engine.device)
    capture = getattr(recorder, 'request_capture', None)
    if hasattr(recorder, 'begin_request'):
        recorder.begin_request()
    started = time.perf_counter()
    from contextlib import nullcontext
    with capture.region('seven_step/request') if capture is not None else nullcontext():
        media = run_two_stage(engine, args, paths)
        torch.cuda.synchronize(engine.device)
        if hasattr(recorder, 'end_request'):
            recorder.end_request()
    if engine.world_size > 1:
        torch.distributed.barrier()
    recorder.timings['pipeline_elapsed_s'] = time.perf_counter() - started
    if media is not None:
        if args.raw_output_path is not None:
            capture_started = time.perf_counter()
            args.raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            with args.raw_output_path.open('xb') as stream:
                torch.save(dict(video=media.video, audio=media.audio,
                    audio_sample_rate=media.audio_sample_rate, duration=5, seed=args.seed,
                    stage_counts=dict(h3=4, ltx=3)), stream)
            recorder.timings['raw_output_capture_s'] = time.perf_counter() - capture_started
        encode_started = time.perf_counter()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        media.save(args.output)
        recorder.timings['media_encoding_s'] = time.perf_counter() - encode_started
        print(json.dumps(dict(output=str(args.output.resolve()), mode=args.mode,
            pipeline_elapsed_s=recorder.timings['pipeline_elapsed_s'],
            media_encoding_s=recorder.timings['media_encoding_s'],
            timing_is_instrumented=args.profile_dir is not None)))
    recorder.success = True


def main(argv=None):
    process_started = time.perf_counter()
    args = parse_args(argv)
    if args.build_prompt_cache:
        from h3_runtime.ltx_cache import build_cache
        build_cache(stage2_paths(args), args.prompt_cache)
        return 0
    from h3_runtime import MiniMaxH3Inference, RuntimeConfig
    config = RuntimeConfig(fsdp_degree=args.fsdp_degree, ulysses_degree=args.ulysses_degree,
        staged_loading=args.staged_loading, flash_sdpa=args.flash_sdpa,
        allow_tf32=args.tf32, allow_bf16_reduced_precision_reduction=args.bf16_reduced_precision_reduction,
        vae_batched=args.vae_batched, vae_global_batch=args.vae_global_batch,
        vae_compile_mode=args.vae_compile_mode, vae_tile_size=args.vae_tile_size,
        adaln_precompute=args.adaln_precompute, kernel_fusions=args.kernel_fusions,
        packed_sp=args.packed_sp, text_fsdp=args.text_fsdp, parallel_vae=args.parallel_vae,
        evidence_dir=None if args.evidence_dir is None else str(args.evidence_dir/'h3'))
    paths = stage2_paths(args) if args.mode == 'two-stage' else None
    if paths is not None:
        for name in ('transformer', 'refiner_lora', 'audio_vae', 'output_video_vae', 'adapter_dir',
                     'h3_upscaler_source', 'h3_upscaler_checkpoint', 'prompt_cache', 'sana_root', 'ltx_root'):
            if not Path(paths[name]).exists():
                raise FileNotFoundError(f'Missing two-stage asset {name}: {paths[name]}')
    references = load_references(args.reference)
    recorder = None
    artifact_dir = args.profile_dir or args.benchmark_dir
    if artifact_dir is not None:
        import torch
        from h3_runtime.profiling import RunInstrumentation, source_hashes, write_json
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        recorder_type = RunInstrumentation
        extra = {}
        if args.mode == 'two-stage':
            from h3_runtime.seven_step_profiling import SevenStepInstrumentation
            recorder_type = SevenStepInstrumentation
            extra = dict(ltx_indices=args.profile_ltx_forward_indices,
                         ltx_attention_backend=args.ltx_attention_backend,
                         profile_scope=args.profile_scope, profile_windows=args.profile_window,
                         record_shapes=args.profile_record_shapes, with_stack=args.profile_with_stack,
                         profile_memory=args.profile_memory, collective_identifiers=args.profile_collective_identifiers)
        recorder = recorder_type(artifact_dir, profile=args.profile_dir is not None,
            rank=int(os.environ.get('RANK', '0')), device=torch.device('cuda', local_rank), torch_module=torch,
            indices=args.profile_forward_indices, flash_sdpa=args.flash_sdpa,
            attention_backend=args.attention_backend, **extra)
    try:
        with MiniMaxH3Inference(str(args.model), args.adapter,
                attention_backend=args.attention_backend, task=args.task,
                reference_image_resize_mode=args.reference_image_resize_mode,
                compute_quant=args.compute_quant, lora_mode=args.lora_mode, runtime=config,
                instrumentation=recorder) as engine:
            if recorder is not None:
                recorder.timings['startup_s'] = time.perf_counter() - process_started
                if engine.is_rank_zero:
                    write_json(artifact_dir/'config.json', dict(
                        arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                        runtime=asdict(config), world_size=engine.world_size,
                        torch_version=torch.__version__, cuda_version=torch.version.cuda,
                        numerical_environment={'DISABLE_ADDMM_CUDA_LT':os.environ.get('DISABLE_ADDMM_CUDA_LT', '0')}))
                    write_json(artifact_dir/'source-sha256.json', source_hashes(Path(__file__).resolve().parent))
                write_json(artifact_dir/f'device-rank{engine.rank}.json', dict(
                    rank=engine.rank, local_rank=local_rank, name=torch.cuda.get_device_name(engine.device),
                    capability=list(torch.cuda.get_device_capability(engine.device)),
                    uuid=str(torch.cuda.get_device_properties(engine.device).uuid),
                    total_memory_bytes=torch.cuda.get_device_properties(engine.device).total_memory))
            if args.evidence_dir is not None and engine.is_rank_zero:
                args.evidence_dir.mkdir(parents=True, exist_ok=True)
                snapshot = dict(arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                                runtime=asdict(config), paths=paths,
                                numerical_environment={'DISABLE_ADDMM_CUDA_LT':os.environ.get('DISABLE_ADDMM_CUDA_LT', '0')})
                (args.evidence_dir/'config.json').write_text(json.dumps(snapshot, indent=2))
                source = Path(__file__).resolve().parent
                hashes = {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(source.rglob('*')) if p.is_file() and p.suffix in {'.py', '.sh'}
                          and not any(part.startswith('.') for part in p.relative_to(source).parts)}
                (args.evidence_dir/'source-sha256.json').write_text(json.dumps(hashes, indent=2))
            if args.mode == 'two-stage':
                if recorder is None:
                    run_two_stage(engine, args, paths)
                else:
                    run_instrumented_two_stage(engine, args, paths, recorder, torch)
            else:
                request = dict(duration=args.duration, seed=args.seed, image=args.image, references=references,
                               num_inference_steps=50 if args.mode == 'original' else 5,
                               width=args.width or 1344, height=args.height or 768)
                if args.warmup:
                    media = engine.generate(args.prompt, **request)
                    del media
                if recorder is not None:
                    if engine.world_size > 1:
                        torch.distributed.barrier()
                    torch.cuda.synchronize(engine.device)
                    started = time.perf_counter()
                media = engine.generate(args.prompt, **request)
                if recorder is not None:
                    torch.cuda.synchronize(engine.device)
                    recorder.timings['pipeline_elapsed_s'] = time.perf_counter() - started
                if media is not None:
                    if args.raw_output_path is not None:
                        # Each benchmark invocation contains one request. Capture is
                        # outside its timer and cannot affect a later timed request.
                        capture_started = time.perf_counter()
                        args.raw_output_path.parent.mkdir(parents=True, exist_ok=True)
                        with args.raw_output_path.open('xb') as stream:
                            torch.save(dict(video=media.video.detach().cpu(),
                                            audio=media.audio.detach().cpu(),
                                            audio_sample_rate=media.audio_sample_rate,
                                            duration=args.duration, seed=args.seed), stream)
                        recorder.timings['raw_output_capture_s'] = time.perf_counter() - capture_started
                    encode_started = time.perf_counter()
                    media.save(args.output)
                    encode_elapsed = time.perf_counter() - encode_started
                    if recorder is not None:
                        recorder.timings['media_encoding_s'] = encode_elapsed
                        recorder.timings['legacy_pipeline_call_s'] = media.elapsed_s
                    # The complete instrumented timer encloses generate(), including request preparation.
                    elapsed = media.elapsed_s if recorder is None else recorder.timings['pipeline_elapsed_s']
                    print(json.dumps(dict(output=str(args.output.resolve()), mode=args.mode,
                                          pipeline_elapsed_s=round(elapsed, 3),
                                          media_encoding_s=round(encode_elapsed, 3),
                                          timing_is_instrumented=(args.profile_dir is not None
                                                                  or args.evidence_dir is not None)), ensure_ascii=False))
                if recorder is not None:
                    recorder.success = True
            if args.evidence_dir is not None:
                from h3_runtime.evidence import capture_resources
                capture_resources(args.evidence_dir, engine.rank, engine.device)
                if engine.is_rank_zero:
                    (args.evidence_dir/'generation-complete.json').write_text(json.dumps(
                        dict(status='GENERATED', numerical_replay='PENDING', mode=args.mode), indent=2))
    finally:
        if recorder is not None:
            recorder.finish(error=sys.exc_info()[1])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
