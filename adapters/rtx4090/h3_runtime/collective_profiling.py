"""Bounded NCCL Flight Recorder snapshots for diagnostic request windows.

record_id is local to one process; collective_seq_id is scoped to a process
 group. Neither is a Chrome correlation id or a CUDA stream dependency.
"""
import hashlib
import json
import os

from .profiling import write_json

BUFFER_ENTRIES = 32768
MAXIMUM_DUMP_BYTES = 64 * 1024 * 1024


def configure_collective_profiling():
    """Call before process groups are initialized, only for explicit captures."""
    os.environ.update(TORCH_NCCL_TRACE_BUFFER_SIZE=str(BUFFER_ENTRIES),
                      TORCH_NCCL_ENABLE_TIMING='1', TORCH_NCCL_TRACE_CPP_STACK='0', TORCH_FR_BUFFER_SIZE=str(BUFFER_ENTRIES), TORCH_FR_CPP_STACK='0')


def checked_dump(payload):
    if not isinstance(payload, (str, bytes)) or len(payload) > MAXIMUM_DUMP_BYTES:
        raise ValueError('Flight Recorder JSON exceeds its bound')
    value = json.loads(payload)
    if not isinstance(value, dict) or not isinstance(value.get('entries', []), list):
        raise ValueError('Invalid Flight Recorder JSON')
    entries = value.get('entries', [])
    if len(entries) > BUFFER_ENTRIES:
        raise ValueError('Flight Recorder entry bound exceeded')
    identifiers = [e.get('record_id') for e in entries]
    if any(type(i) is not int or i < 0 for i in identifiers) or identifiers != sorted(set(identifiers)):
        raise ValueError('Flight Recorder local record ids are invalid')
    return value, identifiers


class CollectiveCapture:
    def __init__(self, torch_module):
        module = torch_module._C._distributed_c10d
        self.dump = getattr(module, '_dump_nccl_trace_json', None)
        if self.dump is None:
            raise RuntimeError('NCCL JSON Flight Recorder API unavailable')
        if (os.environ.get('TORCH_NCCL_TRACE_BUFFER_SIZE') != str(BUFFER_ENTRIES)
                or os.environ.get('TORCH_NCCL_ENABLE_TIMING') != '1'):
            raise RuntimeError('Flight Recorder must be configured before process group creation')
        self.start_ids = None

    def start(self):
        value, identifiers = checked_dump(self.dump(True, False))
        self.start_ids = identifiers
        self.start_version = value.get('version')
        self.start_epoch = value.get('reset_epoch')
        self.start_boundary = ({k:value['entries'][-1].get(k) for k in ('record_id','time_created_ns')} if identifiers else None)

    def stop(self, directory, rank, index, window):
        if self.start_ids is None:
            raise RuntimeError('Flight Recorder window has no start boundary')
        value, identifiers = checked_dump(self.dump(True, False))
        boundary = self.start_ids[-1] if self.start_ids else -1
        selected = [e for e in value.get('entries', []) if e['record_id'] > boundary]
        ids = [e['record_id'] for e in selected]
        retained = next((e for e in value.get('entries', []) if e['record_id']==boundary), None)
        retained_boundary = ({k:retained.get(k) for k in ('record_id','time_created_ns')} if retained else None)
        # A retained boundary also catches resets when the API omits epoch data.
        reset = (value.get('version') != self.start_version
                 or value.get('reset_epoch') != self.start_epoch
                 or self.start_boundary != retained_boundary)
        contiguous = bool(ids) and ids[0] == boundary+1 and ids[-1]-boundary == len(ids)
        complete = not reset and contiguous and len(identifiers) < BUFFER_ENTRIES
        output = dict(schema_version=1, rank=rank, window=window,
                      status='completed' if complete else 'incomplete',
                      buffer_entries=BUFFER_ENTRIES, start_record_id=boundary,
                      start_boundary=self.start_boundary, retained_start_boundary=retained_boundary,
                      end_record_id=identifiers[-1] if identifiers else None,
                      reset_or_missing_boundary=bool(reset), records_contiguous=contiguous,
                      buffer_full=len(identifiers) == BUFFER_ENTRIES,
                      flight_recorder={**value, 'entries': selected})
        path = directory / f'collectives-rank{rank}-window{index}.json'
        write_json(path, output)
        self.start_ids = None
        return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    status=output['status'], entry_count=len(selected))
