"""CPU integration checks for entry selection and distributed launch policy."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def recorder(self, directory):
        interpreter = directory / "python recorder"
        interpreter.write_text(
            "#!/usr/bin/env python3\n"
            "import json,os,sys\n"
            "if len(sys.argv)>1 and sys.argv[1].endswith('/tools/preflight_4090.py'):\n"
            " with open(os.environ['TEST_COMMAND_FILE']+'.preflight','w') as f:\n"
            "  json.dump(sys.argv[2:],f)\n"
            " sys.exit(int(os.environ.get('TEST_PREFLIGHT_EXIT','0')))\n"
            "if sys.argv[1:]==['-']:\n"
            " sys.stdin.read()\n"
            " sys.exit(int(os.environ.get('TEST_GPU_CHECK_EXIT','0')))\n"
            "with open(os.environ['TEST_COMMAND_FILE'],'w') as f:\n"
            " json.dump({'args':sys.argv[1:],'env':{k:os.environ.get(k) for k in "
            "['HF_HUB_OFFLINE','DISABLE_ADDMM_CUDA_LT']}},f)\n"
        )
        interpreter.chmod(0o755)
        output = directory / "command.json"
        env = dict(os.environ, SOL_H3_PYTHON=str(interpreter),
                   SOL_H3_OFFICIAL_PYTHON=str(interpreter), TEST_COMMAND_FILE=str(output))
        env.pop("SOL_H3_FSDP_DEGREE", None)
        env.pop("SOL_H3_OFFICIAL_GPUS", None)
        env.pop("DISABLE_ADDMM_CUDA_LT", None)
        return output, env

    def test_4090_launch_policy_and_prompt_quoting(self):
        with tempfile.TemporaryDirectory(prefix="sol h3 launcher ") as directory:
            output, env = self.recorder(Path(directory))
            prompt = "A dancer with spaces and literal $ symbols."
            result = subprocess.run(
                ["bash", str(ROOT / "run_8x4090.sh"), "--model", "/models/H3",
                 "--adapter", "/models/adapter.safetensors", "--prompt", prompt,
                 "--output", "/outputs/video.mp4"],
                cwd=directory, env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            preflight = json.loads(Path(str(output) + ".preflight").read_text())
            self.assertEqual(preflight[:2], ["--gpu", "--launcher-args"])
            self.assertEqual(preflight[preflight.index("--prompt") + 1], prompt)
            self.assertEqual(preflight[preflight.index("--model") + 1], "/models/H3")
            record = json.loads(output.read_text())
            args = record["args"]
            self.assertEqual(args[:3], ["-m", "torch.distributed.run", "--standalone"])
            self.assertIn("--nproc-per-node=8", args)
            self.assertEqual(Path(args[4]), ROOT / "adapters/rtx4090/infer.py")
            for option, value in [("--fsdp-degree", "8"), ("--ulysses-degree", "8"),
                                  ("--attention-backend", "dense"), ("--mode", "single-stage")]:
                self.assertEqual(args[args.index(option) + 1], value)
            for option in ["--staged-loading", "--text-fsdp", "--flash-sdpa",
                           "--no-tf32", "--no-bf16-reduced-precision-reduction"]:
                self.assertIn(option, args)
            self.assertEqual(args[args.index("--prompt") + 1], prompt)
            self.assertEqual(record["env"]["HF_HUB_OFFLINE"], "1")

    def test_asset_preflight_failure_prevents_worker_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            output, env = self.recorder(Path(directory))
            env["TEST_PREFLIGHT_EXIT"] = "9"
            result = subprocess.run(
                ["bash", str(ROOT / "run_8x4090.sh"), "--model=/models/H3",
                 "--adapter=/models/adapter.safetensors"],
                cwd=directory, env=env, capture_output=True,
            )
            self.assertEqual(result.returncode, 9)
            self.assertTrue(Path(str(output) + ".preflight").exists())
            self.assertFalse(output.exists())

    def test_gpu_precheck_failure_prevents_worker_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            output, env = self.recorder(Path(directory))
            env["TEST_GPU_CHECK_EXIT"] = "7"
            result = subprocess.run(["bash", str(ROOT / "run_8x4090.sh")],
                                    cwd=directory, env=env, capture_output=True)
            self.assertEqual(result.returncode, 7)
            self.assertFalse(output.exists())

    def test_official_dispatch_stays_in_official_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            output, env = self.recorder(Path(directory))
            env["SOL_H3_OFFICIAL_GPUS"] = "2"
            result = subprocess.run(["bash", str(ROOT / "run_official.sh"), "--task", "t2v"],
                                    cwd=directory, env=env, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(output.read_text())["args"]
            self.assertIn("--nproc-per-node=2", args)
            self.assertEqual(Path(args[4]), ROOT / "official/Sol-H3/infer.py")
            output.unlink()
            env["SOL_H3_OFFICIAL_GPUS"] = "3"
            result = subprocess.run(["bash", str(ROOT / "run_official.sh")],
                                    cwd=directory, env=env, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())

    def test_default_entry_selects_official_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            output, env = self.recorder(Path(directory))
            env["SOL_H3_OFFICIAL_GPUS"] = "1"
            result = subprocess.run(
                ["bash", str(ROOT / "run.sh"), "--attention-backend", "dense"],
                cwd=directory, env=env, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(output.read_text())["args"]
            self.assertEqual(Path(args[0]), ROOT / "official/Sol-H3/infer.py")
            self.assertEqual(args[1:], ["--attention-backend", "dense"])

    def test_4090_profile_keeps_automatic_asset_check_and_eight_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            output, env = self.recorder(Path(directory))
            env["SOL_H3_FSDP_DEGREE"] = "1"
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/run_lins_rtx4090_24g_8gpu.sh"),
                 "--model", "/models/H3", "--adapter", "/models/adapter.safetensors"],
                cwd=directory, env=env, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(Path(str(output) + ".preflight").exists())
            args = json.loads(output.read_text())["args"]
            self.assertIn("--nproc-per-node=8", args)
            self.assertEqual(Path(args[4]), ROOT / "adapters/rtx4090/infer.py")
            self.assertEqual(args[args.index("--fsdp-degree") + 1], "8")

    def test_server_profile_help_does_not_require_torch(self):
        env = dict(os.environ, SOL_H3_PYTHON=sys.executable,
                   SOL_H3_OFFICIAL_PYTHON=sys.executable)
        scripts = ["scripts/run_lins_rtx4090_24g_8gpu.sh"]
        for script in scripts:
            with self.subTest(script=script):
                result = subprocess.run(["bash", str(ROOT / script), "--help"],
                                        cwd="/tmp", env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--model", result.stdout)

    def test_adapted_help_has_no_cuda_dependency(self):
        env = dict(os.environ, SOL_H3_PYTHON=sys.executable)
        for script in ["run_8x4090.sh", "run_original_8x4090.sh", "run_two_stage_8x4090.sh"]:
            with self.subTest(script=script):
                result = subprocess.run(["bash", str(ROOT / script), "--help"],
                                        cwd="/tmp", env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--model", result.stdout)
                self.assertIn("--output", result.stdout)


if __name__ == "__main__":
    unittest.main()
