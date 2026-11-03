#!/usr/bin/env python3
"""One persistent FIFO queue around an existing SOL-H3 launch script (Linux)."""
from __future__ import annotations

import argparse
import fcntl
import hmac
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Jobs:
    def __init__(self, directory, command, cwd, capacity=16):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.instance_lock = (self.directory / "server.lock").open("a")
        try:
            fcntl.flock(self.instance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.instance_lock.close()
            raise RuntimeError("Another API process owns this data directory") from None
        self.command, self.cwd, self.capacity = command, str(cwd), capacity
        self.lock = threading.RLock()
        self.pending = queue.Queue()
        self.stop = threading.Event()
        self.process = None
        self.records = {}
        for path in sorted(self.directory.glob("*/job.json")):
            item = json.loads(path.read_text())
            if item["status"] == "running":
                item.update(status="failed", finished_at=time.time(),
                            error="Service interrupted; submit a new request to retry")
                self.save(item)
            self.records[item["id"]] = item
        for item in sorted(self.records.values(), key=lambda x: x["created_at"]):
            if item["status"] == "queued":
                self.pending.put(item["id"])
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def save(self, item):
        folder = self.directory / item["id"]
        folder.mkdir(exist_ok=True)
        temporary = folder / "job.json.tmp"
        temporary.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(folder / "job.json")

    def submit(self, parameters):
        with self.lock:
            if self.stop.is_set():
                raise RuntimeError("Service is stopping")
            if sum(x["status"] in ("queued", "running") for x in self.records.values()) >= self.capacity:
                raise queue.Full
            item = dict(id=uuid.uuid4().hex, status="queued", created_at=time.time(),
                        parameters=parameters, error=None)
            self.save(item)
            self.records[item["id"]] = item
            self.pending.put(item["id"])
            return dict(item)

    def get(self, job_id):
        with self.lock:
            item = self.records.get(job_id)
            if item is None:
                return None
            item = dict(item)
            if item["status"] == "completed":
                item["video_url"] = f"/v1/jobs/{job_id}/video"
            return item

    def run(self):
        while not self.stop.is_set():
            try:
                job_id = self.pending.get(timeout=0.2)
            except queue.Empty:
                continue
            with self.lock:
                if self.stop.is_set():
                    return  # The persisted queued job resumes at next startup.
                item = self.records[job_id]
                folder = self.directory / job_id
                output = folder / "video.mp4"
                args = item["parameters"]
                command = self.command + ["--task", "t2v", "--duration", str(args["duration"]),
                                          "--seed", str(args["seed"]), "--prompt", args["prompt"],
                                          "--output", str(output)]
                item.update(status="running", started_at=time.time())
                self.save(item)
            try:
                with (folder / "inference.log").open("wb") as log:
                    with self.lock:
                        if self.stop.is_set():
                            raise RuntimeError("Service stopped before inference launch")
                        self.process = subprocess.Popen(command, cwd=self.cwd, stdin=subprocess.DEVNULL,
                                                        stdout=log, stderr=subprocess.STDOUT,
                                                        start_new_session=True,
                                                        pass_fds=(self.instance_lock.fileno(),))
                    code = self.process.wait()
                if code != 0:
                    raise RuntimeError(f"Inference exited with code {code}; see task log")
                if not output.is_file() or output.stat().st_size == 0:
                    raise RuntimeError("Inference did not produce a video")
                result = dict(status="completed", video_bytes=output.stat().st_size, error=None)
            except Exception as exc:
                result = dict(status="failed", error=str(exc))
            with self.lock:
                self.process = None
                item.update(result, finished_at=time.time())
                self.save(item)

    def close(self):
        self.stop.set()
        with self.lock:
            process = self.process
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.worker.join(timeout=15)
        self.instance_lock.close()


def parameters(body):
    if not isinstance(body, dict) or set(body) - {"prompt", "seed", "duration"}:
        raise ValueError("Allowed fields: prompt, seed, duration")
    prompt, seed, duration = body.get("prompt"), body.get("seed", 0), body.get("duration", 5)
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000 or "\0" in prompt:
        raise ValueError("prompt must contain 1–4000 characters without NUL")
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("seed must be an integer from 0 to 2147483647")
    if type(duration) is not int or duration != 5:
        raise ValueError("This service currently accepts duration=5 only")
    return dict(prompt=prompt, seed=seed, duration=duration)


def handler(jobs, api_key):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def reply(self, code, value, content_type="application/json"):
            data = (json.dumps(value, ensure_ascii=False).encode() if content_type == "application/json"
                    else value.encode())
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            expected = ("Bearer " + api_key).encode()
            supplied = self.headers.get("Authorization", "").encode()
            if api_key and not hmac.compare_digest(supplied, expected):
                self.reply(401, {"error": "Bearer token required"})
                return False
            return True

        def do_POST(self):
            if not self.authorized():
                return
            if urlsplit(self.path).path != "/v1/jobs":
                return self.reply(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384 or self.headers.get("Transfer-Encoding"):
                    return self.reply(413, {"error": "Use a JSON body with Content-Length <= 16384"})
                body = parameters(json.loads(self.rfile.read(length)))
                item = jobs.submit(body)
            except (ValueError, UnicodeError):
                return self.reply(400, {"error": "Invalid JSON or parameters; require prompt, seed>=0, duration=5"})
            except queue.Full:
                return self.reply(429, {"error": "Queue is full; retry later"})
            except RuntimeError:
                return self.reply(503, {"error": "Service is stopping"})
            self.reply(202, item)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/health":
                ready = jobs.worker.is_alive() and not jobs.stop.is_set()
                return self.reply(200 if ready else 503, {"status": "ok" if ready else "unavailable", "concurrency": 1,
                                        "scope": "API process only; GPU readiness is checked per task"})
            if not self.authorized():
                return
            match = re.fullmatch(r"/v1/jobs/([a-f0-9]{32})(?:/(video|log))?", path)
            if not match:
                return self.reply(404, {"error": "Not found"})
            job_id, resource = match.groups()
            item = jobs.get(job_id)
            if item is None:
                return self.reply(404, {"error": "Job not found"})
            if resource is None:
                return self.reply(200, item)
            folder = jobs.directory / job_id
            if resource == "log":
                log = folder / "inference.log"
                if not log.exists():
                    return self.reply(200, "", "text/plain")
                with log.open("rb") as file:
                    file.seek(max(0, log.stat().st_size - 65536))
                    return self.reply(200, file.read().decode(errors="replace"), "text/plain")
            if item["status"] != "completed":
                return self.reply(409, {"error": "Video is not ready", "status": item["status"]})
            video = folder / "video.mp4"
            if not video.is_file():
                return self.reply(410, {"error": "Video file has been removed"})
            with video.open("rb") as file:
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(video.stat().st_size))
                self.send_header("Content-Disposition", 'attachment; filename="video.mp4"')
                self.end_headers()
                shutil.copyfileobj(file, self.wfile)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--entry", type=Path, default=ROOT / "run.sh")
    parser.add_argument("--model", default=os.environ.get("SOL_H3_MODEL", "/Datasets/public_models/SOL-H3/MiniMax-H3"))
    parser.add_argument("--adapter", default=os.environ.get("SOL_H3_ADAPTER", "/Datasets/public_models/SOL-H3/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors"))
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "sol-h3-api")
    parser.add_argument("--capacity", type=int, default=16, help="Maximum queued plus running tasks")
    parser.add_argument("--launch-arg", action="append", default=[], help="Fixed extra launcher argument; use --launch-arg=--FLAG")
    args = parser.parse_args()
    key = os.environ.get("SOL_H3_API_KEY", "")
    if args.capacity < 1 or not args.entry.resolve().is_file():
        parser.error("Require capacity>=1 and an existing launch script")
    command = ["bash", str(args.entry.resolve()), "--model", args.model, "--adapter", args.adapter] + args.launch_arg
    jobs = Jobs(args.data_dir, command, ROOT, args.capacity)
    server = None
    try:
        server = ThreadingHTTPServer((args.host, args.port), handler(jobs, key))
        def shutdown(signum, frame):
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        print(f"SOL-H3 API listening on {args.host}:{server.server_port}; concurrency=1", flush=True)
        server.serve_forever()
    finally:
        if server is not None:
            server.server_close()
        jobs.close()


if __name__ == "__main__":
    main()
