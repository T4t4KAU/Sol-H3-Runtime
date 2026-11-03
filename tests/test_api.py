"""HTTP and queue lifecycle checks using a CPU-only stand-in launcher."""
import importlib.util
import json
import os
import select
import shlex
import subprocess
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("serve_api", ROOT / "tools/serve_api.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.launch = self.root / "fake.py"
        self.launch.write_text(
            "import sys,time,json\nfrom pathlib import Path\n"
            "args=sys.argv[1:]\n"
            "prompt=args[args.index('--prompt')+1]\n"
            "if prompt=='failure': raise SystemExit(7)\n"
            "if prompt=='slow': time.sleep(60)\n"
            "time.sleep(0.15)\n"
            "Path(args[args.index('--output')+1]).write_bytes(prompt.encode())\n"
        )
        self.jobs = api.Jobs(self.root / "jobs", [sys.executable, str(self.launch)], self.root, 2)
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), api.handler(self.jobs, "secret"))
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.http.server_port}"

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.jobs.close()
        self.temp.cleanup()

    def request(self, path, data=None, authenticated=True):
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = "Bearer secret"
        request = urllib.request.Request(self.url + path, headers=headers,
                                         data=None if data is None else json.dumps(data).encode())
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as response:
            return response.code, response.read()

    def finished(self, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            item = self.jobs.get(job_id)
            if item["status"] in ("completed", "failed"):
                return item
            time.sleep(0.02)
        self.fail("Task did not finish")

    def test_http_auth_fifo_literal_prompt_and_download(self):
        self.assertEqual(self.request("/health", authenticated=False)[0], 200)
        self.assertEqual(self.request("/v1/jobs", {"prompt": "a"}, False)[0], 401)
        prompt = "dance; $(touch should-not-exist) `literal` 中文"
        code, body = self.request("/v1/jobs", {"prompt": prompt})
        self.assertEqual(code, 202)
        first = json.loads(body)["id"]
        code, body = self.request("/v1/jobs", {"prompt": "second"})
        self.assertEqual(code, 202)
        second = json.loads(body)["id"]
        self.assertEqual(self.request("/v1/jobs", {"prompt": "third"})[0], 429)
        self.assertEqual(self.request(f"/v1/jobs/{second}/video")[0], 409)
        one, two = self.finished(first), self.finished(second)
        self.assertEqual(one["status"], "completed")
        self.assertLessEqual(one["finished_at"], two["started_at"])
        self.assertEqual(self.request(f"/v1/jobs/{first}/video"), (200, prompt.encode()))
        self.assertEqual(json.loads(self.request(f"/v1/jobs/{first}")[1])["status"], "completed")
        self.assertEqual(self.request(f"/v1/jobs/{first}/log")[0], 200)
        self.assertFalse((self.root / "should-not-exist").exists())

    def test_cli_non_loopback_without_key(self):
        entry = self.root / "fake.sh"
        entry.write_text(f'exec {shlex.quote(sys.executable)} {shlex.quote(str(self.launch))} "$@"\n')
        env = dict(os.environ, SOL_H3_API_KEY="")
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "tools/serve_api.py"), "--host", "0.0.0.0",
             "--port", "0", "--entry", str(entry), "--data-dir", str(self.root / "keyless")],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertTrue(select.select([process.stdout], [], [], 5)[0], "API startup timed out")
            line = process.stdout.readline()
            self.assertIn("API listening on 0.0.0.0:", line)
            port = int(line.split("0.0.0.0:")[1].split(";")[0])
            self.url = f"http://127.0.0.1:{port}"
            code, body = self.request("/v1/jobs", {"prompt": "keyless"}, False)
            self.assertEqual(code, 202)
            job_id = json.loads(body)["id"]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                code, body = self.request(f"/v1/jobs/{job_id}", authenticated=False)
                self.assertEqual(code, 200)
                if json.loads(body)["status"] == "completed":
                    break
                time.sleep(0.02)
            self.assertEqual(self.request(f"/v1/jobs/{job_id}/video", authenticated=False),
                             (200, b"keyless"))
        finally:
            process.terminate()
            process.communicate(timeout=10)

    def test_resident_load_once_readiness_and_fifo_without_child_process(self):
        self.jobs.close()
        gate = threading.Event()
        loads = []
        closed = []
        class Backend:
            fatal_error = None
            def health(inner):
                return dict(model_load_count=1, model_instance_id="same-engine")
            def generate(inner, parameters, output, log):
                time.sleep(0.1)
                output.write_bytes(parameters["prompt"].encode())
                return dict(model_load_count=1, model_instance_id="same-engine", worker_pid=os.getpid())
            def close(inner):
                closed.append(True)
        def factory():
            gate.wait(5)
            loads.append(True)
            return Backend()
        self.jobs = api.Jobs(self.root / "resident", [], self.root, 2, backend_factory=factory)
        self.http.RequestHandlerClass = api.handler(self.jobs, "")
        self.assertEqual(self.request("/health", authenticated=False)[0], 503)
        gate.set()
        self.assertTrue(self.jobs.ready.wait(5))
        code, body = self.request("/health", authenticated=False)
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["model_load_count"], 1)
        with patch.object(api.subprocess, "Popen", side_effect=AssertionError("Resident request spawned a child")):
            ids = [json.loads(self.request("/v1/jobs", {"prompt": prompt}, False)[1])["id"]
                   for prompt in ("first", "second")]
            records = [self.finished(job_id) for job_id in ids]
        self.assertEqual(loads, [True])
        self.assertLessEqual(records[0]["finished_at"], records[1]["started_at"])
        for item, prompt in zip(records, ("first", "second")):
            self.assertEqual(item["status"], "completed")
            self.assertEqual(item["model_instance_id"], "same-engine")
            self.assertEqual(item["worker_pid"], os.getpid())
            self.assertEqual(self.request(f"/v1/jobs/{item['id']}/video", authenticated=False),
                             (200, prompt.encode()))
        self.jobs.close()
        self.assertEqual(closed, [True])

    def test_resident_startup_failure_is_unready_and_rejects_jobs(self):
        self.jobs.close()
        def factory():
            raise RuntimeError("model load failed")
        self.jobs = api.Jobs(self.root / "bad-resident", [], self.root, backend_factory=factory)
        self.http.RequestHandlerClass = api.handler(self.jobs, "")
        self.jobs.worker.join(timeout=5)
        code, body = self.request("/health", authenticated=False)
        self.assertEqual(code, 503)
        self.assertIn("model load failed", json.loads(body)["error"])
        self.assertEqual(self.request("/v1/jobs", {"prompt": "hello"}, False)[0], 503)

    def test_resident_rejects_subprocess_launch_overrides(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/serve_api.py"),
            "--backend", "h800-resident", "--launch-arg=--no-adaln-precompute"],
            capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 2)
        self.assertIn("resident H800 profile is fixed", result.stderr)

    def test_seven_step_requires_server_configuration(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/serve_api.py"),
            "--backend", "h800-seven-step"], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --seven-step-config", result.stderr)

    def test_validation_failure_and_single_instance(self):
        for body in [{"prompt": ""}, {"prompt": "x", "seed": True},
                     {"prompt": "x", "duration": 10}, {"prompt": "x", "entry": "evil"}]:
            self.assertEqual(self.request("/v1/jobs", body)[0], 400)
        with self.assertRaises(RuntimeError):
            api.Jobs(self.root / "jobs", [], self.root)
        code, body = self.request("/v1/jobs", {"prompt": "failure"})
        self.assertEqual(code, 202)
        item = self.finished(json.loads(body)["id"])
        self.assertEqual(item["status"], "failed")
        self.assertIn("code 7", item["error"])
        self.assertEqual(self.request("/v1/jobs/" + "0" * 32)[0], 404)

    def test_restart_recovers_queue_and_marks_interrupted_task(self):
        self.jobs.close()
        directory = self.root / "recovery"
        for i, state in enumerate(["running", "queued"]):
            job_id = str(i) * 32
            folder = directory / job_id
            folder.mkdir(parents=True)
            (folder / "job.json").write_text(json.dumps(dict(
                id=job_id, status=state, created_at=i,
                parameters=dict(prompt="recovered", seed=0, duration=5), error=None)))
        jobs = api.Jobs(directory, [sys.executable, str(self.launch)], self.root)
        try:
            self.assertEqual(jobs.get("0" * 32)["status"], "failed")
            deadline = time.monotonic() + 5
            while jobs.get("1" * 32)["status"] != "completed" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(jobs.get("1" * 32)["status"], "completed")
        finally:
            jobs.close()

    def test_stop_terminates_active_process_and_preserves_pending_job(self):
        first = self.jobs.submit(api.parameters({"prompt": "slow"}))["id"]
        second = self.jobs.submit(api.parameters({"prompt": "next"}))["id"]
        deadline = time.monotonic() + 5
        while self.jobs.process is None and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertIsNotNone(self.jobs.process)
        child = self.jobs.process
        self.jobs.close()
        self.assertIsNotNone(child.poll())
        self.assertEqual(self.jobs.get(first)["status"], "failed")
        self.assertEqual(self.jobs.get(second)["status"], "queued")
        restarted = api.Jobs(self.root / "jobs", [sys.executable, str(self.launch)], self.root)
        try:
            deadline = time.monotonic() + 5
            while restarted.get(second)["status"] != "completed" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(restarted.get(second)["status"], "completed")
        finally:
            restarted.close()


if __name__ == "__main__":
    unittest.main()
