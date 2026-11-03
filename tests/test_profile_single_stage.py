"""CPU checks for observation scope; these do not validate CUDA profiling."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProfileArgumentsTests(unittest.TestCase):
    def run_check(self, before=(), after=()):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model").mkdir()
            (path / "adapter").touch()
            return subprocess.run(
                [sys.executable, str(ROOT / "tools/profile_single_stage.py"),
                 "--artifacts", str(path / "capture"), "--check-arguments", *before, "--",
                 "--model", str(path / "model"), "--adapter", str(path / "adapter"),
                 "--output", str(path / "video.mp4"), "--prompt", "A boat", "--staged-loading", *after],
                capture_output=True, text=True)

    def test_four_forward_scope_without_cuda(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value["expected_forwards"], 4)
        self.assertFalse(value["cuda_executed"])
        self.assertEqual(value["forward_indices"], [0, 2, 3])

    def test_rejects_49_step_indices_and_warmup(self):
        self.assertNotEqual(self.run_check(before=("--forward-indices", "0,24,48")).returncode, 0)
        self.assertNotEqual(self.run_check(after=("--warmup",)).returncode, 0)

    def test_rejects_combined_legacy_profiler(self):
        self.assertNotEqual(self.run_check(after=("--profile-dir", "/tmp/unused-profile")).returncode, 0)


if __name__ == "__main__":
    unittest.main()
