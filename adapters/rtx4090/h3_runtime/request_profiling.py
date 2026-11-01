"""One profiler owner for a seven-step request or explicitly selected regions."""
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import socket
import time

from .profiling import summarize_trace, write_json
from .python_call_profiling import PythonCallCapture
from .collective_profiling import CollectiveCapture


def audit_forward_attention(events, required_regions):
    """Attribute math SDPA to captured Flash-only forwards on their CPU thread."""
    regions = [e for e in events if e.get('ph') == 'X'
               and e.get('cat') == 'user_annotation' and e.get('name') in required_regions
               and all(k in e for k in ('pid', 'tid', 'ts', 'dur'))]
    violations = []
    for event in events:
        if (event.get('ph') != 'X' or event.get('cat') != 'cpu_op'
                or '_scaled_dot_product_attention_math' not in event.get('name', '')):
            continue
        for region in regions:
            if (event.get('pid') == region['pid'] and event.get('tid') == region['tid']
                    and region['ts'] <= event.get('ts', -1) < region['ts'] + region['dur']):
                violations.append(dict(region=region['name'], operator=event['name']))
    return dict(checked_regions=sorted({e['name'] for e in regions}),
                missing_regions=sorted(set(required_regions)-{e['name'] for e in regions}),
                math_sdpa_violations=violations,
                scope='Captured Flash-only forward CPU regions; absence of math SDPA '
                      'does not independently prove a particular CUDA kernel backend.')


class RequestCapture:
    def __init__(self, directory, *, torch_module, rank, device, scope,
                 windows=(), record_shapes=False, with_stack=False, profile_memory=False,
                 flash_labels=(), collective_identifiers=False):
        self.directory, self.torch = Path(directory), torch_module
        self.rank, self.device, self.scope = rank, device, scope
        self.windows = tuple(windows)
        self.flash_labels = tuple(flash_labels)
        self.required_regions = set()
        self.options = dict(record_shapes=record_shapes, with_stack=with_stack,
                            profile_memory=profile_memory)
        self.collectives = CollectiveCapture(torch_module) if collective_identifiers else None
        self.active = None
        self.python_calls = None
        self.completed = []
        self.spans = []
        self.stack = []
        self.error = None
        self.request_completed = False
        self.request_peaks = None
        self.startup_peaks = None
        self.sequence = 0
        self.clock = dict(hostname=socket.gethostname(),
                          boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                          clock='CLOCK_MONOTONIC', pid=os.getpid())
        self.anchor = dict(monotonic_ns=time.monotonic_ns(), unix_ns=time.time_ns())

    def start(self, name):
        if self.active is not None:
            raise RuntimeError('Nested profiler sessions are forbidden')
        # Native with_stack retains nn.Module instances on the deployed runtime,
        # preventing staged model release. Use source-only annotations instead.
        options = dict(self.options, with_stack=False)
        profiler = self.torch.profiler.profile(
            activities=[self.torch.profiler.ProfilerActivity.CPU,
                        self.torch.profiler.ProfilerActivity.CUDA], **options)
        if self.collectives is not None:
            self.collectives.start()
        profiler.__enter__()
        self.active = (name, profiler)
        self.required_regions = set()
        if self.options['with_stack']:
            self.python_calls = PythonCallCapture(self.torch.profiler.record_function)
            try:
                self.python_calls.start()
            except BaseException:
                self.python_calls = None
                self.active = None
                profiler.__exit__(None, None, None)
                raise

    def stop(self):
        if self.active is None:
            return
        name, profiler = self.active
        self.active = None
        python_source = self.python_calls.stop() if self.python_calls is not None else None
        self.python_calls = None
        profiler.__exit__(None, None, None)
        trace = self.directory / f'trace-request-rank{self.rank}-window{len(self.completed)}.json'
        profiler.export_chrome_trace(str(trace))
        raw_events = json.loads(trace.read_text()).get('traceEvents', [])
        events = summarize_trace(trace, events=raw_events)
        attention = audit_forward_attention(raw_events, self.required_regions)
        kernels = sum(e['count'] for e in events if e['category'] == 'kernel')
        if (name == 'seven_step/request' or '/forward/' in name) and not kernels:
            raise RuntimeError('Requested GPU window contains no CUDA kernel events')
        with trace.open('rb') as stream:
            trace_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        collectives = (self.collectives.stop(self.directory, self.rank, len(self.completed), name)
                       if self.collectives is not None else None)
        self.completed.append(dict(name=name, trace=trace.name, sha256=trace_hash, gpu_kernel_events=kernels,
                                   gpu_events=events, attention=attention, python_source=python_source,
                                   **({'collectives': collectives} if collectives is not None else {})))
        if collectives is not None and collectives['status'] != 'completed':
            raise RuntimeError('Collective capture is incomplete; inspect the retained diagnostic sidecar')
        if attention['missing_regions']:
            raise RuntimeError('Missing Flash-only forward markers in trace')
        if attention['math_sdpa_violations']:
            raise RuntimeError('DiT trace contains math SDPA despite requested Flash-only attention')

    @contextmanager
    def region(self, name):
        identifier = self.sequence
        self.sequence += 1
        parent = self.stack[-1] if self.stack else None
        selected = ((self.scope == 'request' and name == 'seven_step/request')
                    or (self.scope == 'windows' and name in self.windows))
        owns = selected and self.active is None
        start = time.monotonic_ns()
        self.stack.append(identifier)
        status = 'completed'
        try:
            if owns:
                self.start(name)
            if self.active is not None and any(name.startswith(label+'/forward/')
                                               for label in self.flash_labels):
                self.required_regions.add(name)
            marker = (self.torch.profiler.record_function(name)
                      if self.active is not None else nullcontext())
            with marker:
                yield
        except BaseException as exc:
            status = 'failed'
            self.error = repr(exc)
            raise
        finally:
            end = time.monotonic_ns()
            self.stack.remove(identifier)
            self.spans.append(dict(span_id=identifier, parent_span_id=parent, name=name,
                                   start_monotonic_ns=start, end_monotonic_ns=end,
                                   host_wall_s=(end-start)/1e9, status=status))
            if owns:
                try:
                    self.stop()
                except BaseException as exc:
                    self.error = repr(exc)
                    if status == 'completed':
                        raise

    def begin_request(self):
        cuda = self.torch.cuda
        self.startup_peaks = dict(allocated_bytes=cuda.max_memory_allocated(self.device),
                                 reserved_bytes=cuda.max_memory_reserved(self.device))
        cuda.reset_peak_memory_stats(self.device)

    def end_request(self):
        cuda = self.torch.cuda
        self.request_peaks = dict(allocated_bytes=cuda.max_memory_allocated(self.device),
                                 reserved_bytes=cuda.max_memory_reserved(self.device))
        self.request_completed = True

    def finish(self, error=None):
        if self.active is not None:
            self.error = 'Unclosed profiling window'
            try:
                self.stop()
            except BaseException as exc:
                self.error = repr(exc)
        observed = {window['name'] for window in self.completed}
        missing = sorted(set(self.windows) - observed) if self.scope == 'windows' else []
        complete = self.request_completed and not (error or self.error or missing)
        result = dict(schema_version=1, rank=self.rank, scope=self.scope, options=self.options,
                      status='completed' if complete else 'failed', clock=self.clock,
                      clock_anchor=self.anchor, windows=self.completed, spans=self.spans,
                      missing_windows=missing, request_allocator_peak=self.request_peaks,
                      startup_allocator_peak=self.startup_peaks,
                      error=str(error) if error else self.error,
                      timing_semantics='Host spans are unsynchronized and may nest. '
                      'Profiler start/stop/export can perturb this diagnostic run.')
        write_json(self.directory / f'request-profile-rank{self.rank}.json', result)
        return result
