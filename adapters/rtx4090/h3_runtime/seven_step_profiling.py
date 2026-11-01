"""Separate H3/LTX stage evidence for the fixed four-plus-three request."""
from __future__ import annotations

from .profiling import RunInstrumentation, write_json
from .request_profiling import RequestCapture


class SevenStepInstrumentation:
    """Compose existing capture logic without treating seven forwards as one model.

    H3's staged loader uses this object's delegated methods. The LTX adapter
    explicitly selects ``stages['ltx']`` so a skipped/duplicated stage cannot
    satisfy a total-forward-count check.
    """
    expected_forwards = 4
    label = 'seven_step/h3'

    def __init__(self, directory, *, profile, rank, device, torch_module,
                 indices=(0, 2, 3), ltx_indices=(0, 1, 2), flash_sdpa=False,
                 attention_backend='dense', ltx_attention_backend='efficient',
                 profile_scope='forwards', profile_windows=(), record_shapes=False,
                 with_stack=False, profile_memory=False, collective_identifiers=False):
        self.stages = {
            'h3': RunInstrumentation(directory, profile=profile, rank=rank, device=device,
                torch_module=torch_module, indices=indices, flash_sdpa=flash_sdpa,
                attention_backend=attention_backend, expected_forwards=4,
                label=self.label, artifact_prefix='h3-'),
            'ltx': RunInstrumentation(directory, profile=profile, rank=rank, device=device,
                torch_module=torch_module, indices=ltx_indices,
                flash_sdpa=ltx_attention_backend == 'flash', attention_backend='dense',
                expected_forwards=3, label='seven_step/ltx', artifact_prefix='ltx-'),
        }
        self.stages['ltx'].attention['configured_sdpa_backend'] = ltx_attention_backend
        self.directory = self.stages['h3'].directory
        self.profile, self.rank = profile, rank
        self.timings = {}
        self.success = self.finalized = False
        self.startup_allocator_peak = self.request_allocator_peak = None
        self.torch, self.device = torch_module, device
        self.request_capture = None
        if profile and profile_scope != 'forwards':
            self.request_capture = RequestCapture(
                directory, torch_module=torch_module, rank=rank, device=device,
                scope=profile_scope, windows=profile_windows, record_shapes=record_shapes,
                with_stack=with_stack, profile_memory=profile_memory,
                collective_identifiers=collective_identifiers,
                flash_labels=[s.label for s in self.stages.values() if s.flash_sdpa])
            for stage in self.stages.values():
                stage.request_capture = self.request_capture
                stage.attention['verification_scope'] = 'captured forward regions only; see request windows'

    def phase(self, name):
        return self.stages['h3'].phase(name)

    def attach(self, model):
        return self.stages['h3'].attach(model)

    def fingerprint(self, name, tensor):
        return self.stages['h3'].fingerprint(name, tensor)

    def audit_attention(self, sparse):
        return self.stages['h3'].audit_attention(sparse)

    def begin_request(self):
        cuda = self.torch.cuda
        self.startup_allocator_peak = dict(allocated_bytes=cuda.max_memory_allocated(self.device),
                                          reserved_bytes=cuda.max_memory_reserved(self.device))
        cuda.reset_peak_memory_stats(self.device)
        for stage in self.stages.values():
            stage.cuda_peak_scope = ('shared device allocator since request entry; includes both stages '
                                     'and output processing, not a stage-specific peak')
        if self.request_capture is not None:
            self.request_capture.startup_peaks = self.startup_allocator_peak

    def end_request(self):
        cuda = self.torch.cuda
        self.request_allocator_peak = dict(allocated_bytes=cuda.max_memory_allocated(self.device),
                                          reserved_bytes=cuda.max_memory_reserved(self.device))
        if self.request_capture is not None:
            self.request_capture.request_peaks = self.request_allocator_peak
            self.request_capture.request_completed = True

    def finish(self, error=None):
        if self.finalized:
            return
        failures = []
        for name, stage in self.stages.items():
            stage.success = self.success
            try:
                stage.finish(error=error)
            except Exception as exc:
                failures.append(f'{name}: {exc}')
        capture_summary = None
        if self.request_capture is not None:
            capture_summary = self.request_capture.finish(error)
            if capture_summary['status'] != 'completed':
                failures.append('request profiling incomplete')
        complete = (self.success and error is None and not failures
                    and all(stage.summary['status'] == 'completed' for stage in self.stages.values()))
        summary = dict(schema_version=2, study='seven_step', rank=self.rank,
            mode='profile' if self.profile else 'benchmark', status='completed' if complete else 'failed',
            expected_forwards={name: stage.expected_forwards for name, stage in self.stages.items()},
            attempted_forwards={name: stage.attempted_forwards for name, stage in self.stages.items()},
            completed_forwards={name: stage.completed_forwards for name, stage in self.stages.items()},
            stages={name: stage.summary for name, stage in self.stages.items()},
            timings=self.timings, timing_is_instrumented=self.profile,
            request_profile=capture_summary,
            request_allocator_peak=self.request_allocator_peak,
            startup_allocator_peak=self.startup_allocator_peak,
            pipeline_timing_semantics='Synchronized complete H3/bridge/LTX request including per-request '
                'component loads, text encoding, denoising, H3 audio decode, LTX video decode and CPU '
                'raw output packaging; excludes startup, raw file saving and media encoding.',
            phase_timing_semantics='Unsynchronized host wall time; nested phases must not be summed.',
            audio_provenance='Released pipeline muxes original H3 PCM; LTX audio latents condition video.',
            cuda_max_allocated_bytes=max(
                (self.startup_allocator_peak or {}).get('allocated_bytes', 0),
                *(stage.summary['cuda_max_allocated_bytes'] for stage in self.stages.values())),
            cuda_max_reserved_bytes=max(
                (self.startup_allocator_peak or {}).get('reserved_bytes', 0),
                *(stage.summary['cuda_max_reserved_bytes'] for stage in self.stages.values())),
            error=None if error is None else str(error), cleanup_errors=failures)
        write_json(self.directory / f'summary-rank{self.rank}.json', summary)
        self.finalized = True
        if error is None and not complete:
            raise RuntimeError('seven_step instrumentation incomplete; see per-rank summary')
