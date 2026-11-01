"""Launch local distributed ranks in the supervisor's process group.

Unlike torchrun workers, these ranks do not create separate sessions. The outer
budget supervisor can therefore terminate every rank with one process-group kill.
"""
import argparse
import os
import signal
import socket
import subprocess
import sys
import time


def launch(command, count, check_gpu=False):
    with socket.socket() as address:
        address.bind(('127.0.0.1', 0))
        port = address.getsockname()[1]
    processes = []
    def interrupted(signum, frame):
        raise InterruptedError(f'Signal {signum}')
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        # Run hardware preflight in this supervised process, not in a shell
        # child that could outlive the shell on an internal request timeout.
        if check_gpu:
            import torch
            if torch.cuda.device_count()!=count:raise ValueError('Visible GPU count differs')
            for index in range(count):
                gpu=torch.cuda.get_device_properties(index)
                if '4090' not in gpu.name or (gpu.major,gpu.minor)!=(8,9):
                    raise ValueError('Expected RTX 4090 (SM89)')
        for rank in range(count):
            env = dict(os.environ, MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
                       RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(count), LOCAL_WORLD_SIZE=str(count))
            processes.append(subprocess.Popen(command, env=env))
        while True:
            statuses = [p.poll() for p in processes]
            if any(code is not None and code != 0 for code in statuses):
                raise RuntimeError('Distributed rank failed: '+repr(statuses))
            if all(code == 0 for code in statuses):
                return
            time.sleep(.05)
    finally:
        # Ignore repeat graceful signals while cleaning up; the outer supervisor
        # still owns a bounded final SIGKILL of this entire group.
        for s in previous:signal.signal(s, signal.SIG_IGN)
        for p in processes:
            if p.poll() is None:p.terminate()
        deadline = time.monotonic()+5
        for p in processes:
            try:p.wait(timeout=max(.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                p.kill();p.wait()
        for s, handler in previous.items():signal.signal(s, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nproc-per-node', type=int, choices=(4, 8), required=True)
    parser.add_argument('--check-4090',action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:parser.error('Rank entry script required')
    launch([sys.executable, *args.command], args.nproc_per_node,args.check_4090)
