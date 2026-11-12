import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.config import load_settings
from src.prompt_loader import load_prompt


class ResourceTests(unittest.TestCase):
    def test_prompts_and_schema_available_outside_project_directory(self):
        previous = Path.cwd()
        with TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for name in ("solver", "extractor", "screen-extractor"):
                    self.assertTrue(load_prompt(name))
                schema = Path(__file__).resolve().parents[1] / "src/schemas/extractor.schema.json"
                self.assertIn("condition", json.loads(schema.read_text(encoding="utf-8"))["properties"])
            finally:
                os.chdir(previous)

    def test_dotenv_stays_in_project_root(self):
        with patch("src.config.dotenv_values", return_value={}) as read:
            load_settings()
        read.assert_called_once_with(Path(__file__).resolve().parents[1] / ".env")
