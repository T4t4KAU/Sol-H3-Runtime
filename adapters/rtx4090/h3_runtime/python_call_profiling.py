"""Source annotations without the native profiler's retained nn.Module objects.

Only frame ids and record_function handles survive a callback. No frame, tensor,
local variables or model objects are stored. Captures the calling thread only.
"""
import sys


class PythonCallCapture:
    def __init__(self, record_function, maximum_calls=100000):
        self.record_function = record_function
        self.maximum_calls = maximum_calls
        self.active = {}
        self.calls = 0
        self.truncated = False
        self.previous = None

    def callback(self, frame, event, arg):
        key = id(frame)
        if event == 'return':
            marker = self.active.pop(key, None)
            if marker is not None:
                marker.__exit__(None, None, None)
        elif event == 'call':
            code = frame.f_code
            filename = code.co_filename.replace('\\', '/')
            # Profiling infrastructure must not recursively instrument itself.
            if ('profiling' in filename or not any(part in filename for part in
                    ('/h3_runtime/', '/diffusers/models/', '/diffusers/modular_pipelines/',
                     '/ltx_core/', '/sana/'))):
                return
            if self.calls >= self.maximum_calls:
                self.truncated = True
                return
            marker = self.record_function(f'python_source::{filename}:{code.co_firstlineno}:{code.co_name}')
            marker.__enter__()
            self.active[key] = marker
            self.calls += 1

    def start(self):
        self.previous = sys.getprofile()
        if self.previous is not None:
            raise RuntimeError('A Python profiling callback is already installed')
        sys.setprofile(self.callback)

    def stop(self):
        sys.setprofile(self.previous)
        for marker in reversed(list(self.active.values())):
            marker.__exit__(None, None, None)
        self.active.clear()
        return dict(implementation='python_source_record_function', calls=self.calls,
                    truncated=self.truncated, scope='Selected model source paths on the calling thread; no native Python stack tracer')
