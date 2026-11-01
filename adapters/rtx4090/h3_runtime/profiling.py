"""Bounded original49 diagnostics, deliberately separate from tensor evidence.

Benchmark mode never constructs a profiler, adds record_function regions, or
copies tensors. Phase durations are unsynchronized host wall times; only the
complete request is synchronized. Trace summaries are deterministic reductions
of recorded events, not estimates of total request time.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def source_hashes(root):
    root = Path(root).resolve()
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*'))
            if path.is_file() and path.suffix in {'.py', '.sh'}
            and not any(part.startswith('.') for part in path.relative_to(root).parts)}


def summarize_trace(path, *, events=None):
    """Sum actual GPU event durations by exact name; overlap is not subtracted."""
    if events is None:
        events = json.loads(Path(path).read_text(encoding='utf-8')).get('traceEvents', [])
    groups = defaultdict(lambda: {'count': 0, 'duration_us': 0.0})
    for event in events:
        category = event.get('cat', '')
        if event.get('ph') != 'X' or category not in {'kernel', 'gpu_memcpy', 'gpu_memset'}:
            continue
        entry = groups[(category, event.get('name', ''))]
        entry['count'] += 1
        entry['duration_us'] += float(event.get('dur', 0.0))
    return [{'category': category, 'name': name, **groups[(category, name)]}
            for category, name in sorted(groups)]


class ForwardCapture:
    """Count completed forwards and profile only selected zero-based indices."""
    def __init__(self, model, recorder):
        self.recorder = recorder
        self.count = 0
        self.profiler = self.region = None
        self.handles = []
        try:
            # Installed after FSDP: begin capture before its root pre-forward
            # hook so unsharding and the associated collectives are included.
            self.handles.append(model.register_forward_pre_hook(self.before, prepend=True))
            self.handles.append(model.register_forward_hook(self.after))
        except BaseException:
            self.close()
            raise

    def before(self, _model, _args):
        r = self.recorder
        r.attempted_forwards += 1
        if r.request_capture is not None:
            self.region = r.request_capture.region(f'{r.label}/forward/{self.count}')
            self.region.__enter__()
            return
        if not r.profile or self.count not in r.indices:
            return
        torch = r.torch
        try:
            self._profile_stopped = False
            self._profile_entered = False
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False, with_stack=False)
            self.profiler.__enter__()
            self._profile_entered = True
            self.region = torch.profiler.record_function(f'{r.label}/forward/{self.count}')
            self.region.__enter__()
        except Exception as exc:
            try:
                self.close()
            except Exception:
                pass  # Preserve the actionable profiler startup error.
            raise RuntimeError(f'Could not start CPU/CUDA {r.label} profiling; check CUPTI availability '
                               'and CUDA profiling permissions. No benchmark result was produced.') from exc

    def after(self, _model, _args, _output):
        index = self.count
        self.count += 1
        self.recorder.completed_forwards = self.count
        if self.recorder.request_capture is not None:
            if self.region is not None:
                region, self.region = self.region, None
                region.__exit__(None, None, None)
            return
        if self.profiler is None:
            return
        self._stop_profile()
        profiler, self.profiler = self.profiler, None
        r = self.recorder
        trace = r.directory / f'trace-{r.artifact_prefix}rank{r.rank}-forward{index}.json'
        profiler.export_chrome_trace(str(trace))
        operators = []
        for event in profiler.key_averages():
            operators.append({name: getattr(event, name, None) for name in (
                'key', 'count', 'cpu_time_total', 'self_cpu_time_total',
                'device_time_total', 'self_device_time_total')})
        operators.sort(key=lambda item: item['key'])
        gpu_events = summarize_trace(trace)
        math_ops = [event['key'] for event in operators
                    if '_scaled_dot_product_attention_math' in event['key']]
        summary = dict(rank=r.rank, forward_index=index, trace=trace.name,
                       duration_unit='microseconds', operators=operators, gpu_events=gpu_events,
                       gpu_duration_semantics='Sum of recorded durations; concurrent events can overlap.',
                       observed_math_attention_operators=math_ops)
        write_json(r.directory / f'operators-{r.artifact_prefix}rank{r.rank}-forward{index}.json', summary)
        r.windows.append(dict(forward_index=index, trace=trace.name,
                              gpu_kernel_events=sum(e['count'] for e in gpu_events if e['category'] == 'kernel'),
                              observed_math_attention_operators=math_ops))
        if not any(event['category'] == 'kernel' for event in gpu_events):
            raise RuntimeError('CUDA profiler produced no kernel events; check CUPTI availability '
                               'and CUDA profiling permissions. CPU-only traces are not accepted.')
        if math_ops and r.flash_sdpa:
            raise RuntimeError('DiT trace contains math SDPA despite requested Flash-only attention')

    def _stop_profile(self):
        if self.region is not None:
            region, self.region = self.region, None
            if self.recorder.request_capture is not None:
                error = RuntimeError('Forward did not return before capture cleanup')
                region.__exit__(type(error), error, None)
            else:
                region.__exit__(None, None, None)
        if (self.profiler is not None and getattr(self, '_profile_entered', False)
                and not getattr(self, '_profile_stopped', False)):
            # Mark before exit so cleanup never attempts to exit it twice.
            self._profile_stopped = True
            self.profiler.__exit__(None, None, None)

    def close(self, expected=None):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        try:
            self._stop_profile()
        finally:
            self.profiler = self.region = None
        if expected is not None and self.count != expected:
            raise RuntimeError(f'{self.recorder.label} expected {expected} forwards, observed {self.count}')


class RunInstrumentation:
    def __init__(self, directory, *, profile, rank, device, torch_module,
                 indices=(0, 24, 48), flash_sdpa=False, attention_backend='dense',
                 expected_forwards=49, label='original49', artifact_prefix=''):
        self.expected_forwards, self.label, self.artifact_prefix = expected_forwards, label, artifact_prefix
        if (not isinstance(expected_forwards, int) or expected_forwards < 1
                or len(set(indices)) != len(indices)
                or any(type(index) is not int or not 0 <= index < expected_forwards for index in indices)):
            raise ValueError('Invalid forward count or profiling indices')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.profile, self.rank, self.device, self.torch = profile, rank, device, torch_module
        self.indices, self.flash_sdpa, self.attention_backend = tuple(indices), flash_sdpa, attention_backend
        self.phases, self.windows, self.fingerprints = [], [], {}
        self.attempted_forwards = self.completed_forwards = 0
        self.timings = {}
        self.capture = None
        self.request_capture = None
        self.finalized = False
        self.success = False
        self.cuda_peak_scope = 'since instrumentation initialization; includes startup and encoding'
        self.attention = {'configured_backend': attention_backend,
                          'verification_scope': 'selected forwards only' if profile else 'not traced'}
        torch_module.cuda.reset_peak_memory_stats(device)

    @contextmanager
    def phase(self, name):
        started = time.perf_counter()
        monotonic_start = time.monotonic_ns()
        # These become visible inside a selected forward's active trace only.
        region = (self.request_capture.region(self.label + '/' + name)
                  if self.request_capture is not None else
                  self.torch.profiler.record_function(self.label + '/' + name) if self.profile else nullcontext())
        status = 'completed'
        try:
            with region:
                try:
                    yield
                except BaseException:
                    if self.request_capture is not None and self.capture is not None:
                        self.capture.close()
                    raise
        except BaseException:
            status = 'failed'
            raise
        finally:
            self.phases.append(dict(name=name, host_wall_s=time.perf_counter() - started, status=status,
                                    start_monotonic_ns=monotonic_start,
                                    end_monotonic_ns=time.monotonic_ns()))

    def attach(self, model):
        if self.capture is not None:
            raise RuntimeError('A request may attach each stage exactly once')
        self.capture = ForwardCapture(model, self)
        return self.capture

    def fingerprint(self, name, tensor):
        if not self.profile or self.request_capture is not None:
            return
        value = tensor.detach().cpu().contiguous().reshape(-1).view(self.torch.uint8)
        self.fingerprints[name] = dict(sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                                       shape=list(tensor.shape), dtype=str(tensor.dtype))

    def audit_attention(self, sparse):
        if sparse is None:
            if self.attention_backend != 'dense':
                raise RuntimeError('Sparse attention requested but no sparse attention observer exists')
            return
        self.attention.update(sparse_calls=sparse.sparse_calls, dense_calls=sparse.dense_calls,
                              declined=dict(sparse.declined))
        if self.attention_backend != 'dense' and not sparse.sparse_calls:
            raise RuntimeError('Sparse attention requested but no sparse calls were observed')

    def finish(self, error=None):
        if self.finalized:
            return
        cleanup_error = None
        try:
            if self.capture is not None:
                self.capture.close()
        except Exception as exc:
            cleanup_error = str(exc)
        expected_windows = list(self.indices) if self.profile and self.request_capture is None else []
        complete = (self.success and error is None and cleanup_error is None
                    and self.completed_forwards == self.expected_forwards
                    and self.attempted_forwards == self.expected_forwards
                    and sorted(w['forward_index'] for w in self.windows) == expected_windows)
        summary = dict(schema_version=1, rank=self.rank, mode='profile' if self.profile else 'benchmark',
                       status='completed' if complete else 'failed',
                       expected_forwards=self.expected_forwards, attempted_forwards=self.attempted_forwards,
                       completed_forwards=self.completed_forwards, selected_forward_indices=expected_windows,
                       windows=self.windows, timings=self.timings, phases=self.phases,
                       phase_timing_semantics='Unsynchronized host wall time; nested phases must not be summed.',
                       pipeline_timing_semantics='Synchronized generate request including preparation, component '
                           'loads, denoising, decoding and raw output packaging; excludes startup and media encoding.',
                       timing_is_instrumented=self.profile, fingerprints=self.fingerprints,
                       fingerprint_scope=('disabled for request/window profiling' if self.request_capture
                                          else 'initial/final denoising latents; exact bytes; profile only'),
                       attention=self.attention, error=None if error is None else str(error),
                       cleanup_error=cleanup_error,
                       cuda_peak_scope=self.cuda_peak_scope,
                       cuda_max_allocated_bytes=self.torch.cuda.max_memory_allocated(self.device),
                       cuda_max_reserved_bytes=self.torch.cuda.max_memory_reserved(self.device))
        write_json(self.directory / f'summary-{self.artifact_prefix}rank{self.rank}.json', summary)
        self.summary = summary
        self.finalized = True
        if error is None and not complete:
            raise RuntimeError(f'{self.label} instrumentation incomplete; see per-rank summary')
