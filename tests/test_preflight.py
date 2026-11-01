"""Check startup validation against the actual modular H3 checkpoint layout."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class PreflightTests(unittest.TestCase):
    def run_check(self, index_name):
        spec = importlib.util.spec_from_file_location("preflight_under_test", ROOT / "tools/preflight_4090.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"verify_sources": types.SimpleNamespace(verify=lambda root: {"status": "passed"})}):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            for component in ["transformer", "text_encoder", "tokenizer", "processor", "scheduler", "audio_scheduler", "vae", "audio_vae"]:
                (model / component).mkdir(parents=True)
            if index_name:
                (model / index_name).write_text("{}")
            adapter = Path(directory) / "adapter.safetensors"
            adapter.touch()
            output = io.StringIO()
            with patch.object(sys, "argv", ["preflight", "--model", str(model), "--adapter", str(adapter)]), patch.object(module.importlib.metadata, "version", return_value="2.11.0"), redirect_stdout(output):
                code = module.main()
            return code, json.loads(output.getvalue())

    def test_modular_and_standard_model_indexes_are_accepted(self):
        for name in ["modular_model_index.json", "model_index.json"]:
            with self.subTest(name=name):
                code, result = self.run_check(name)
                self.assertEqual(code, 0, result["errors"])

    def test_missing_model_index_is_rejected(self):
        code, result = self.run_check(None)
        self.assertEqual(code, 1)
        self.assertIn("model_index.json or modular_model_index.json", result["errors"][0]["missing_model_components"])


if __name__ == "__main__":
    unittest.main()
