from __future__ import annotations

import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

if find_spec("typer") is not None:
    from typer.testing import CliRunner
    from continuum.cli import app
else:
    CliRunner = None
    app = None


@unittest.skipIf(find_spec("typer") is None, "typer is not installed in this interpreter")
class TestAccelerateCli(unittest.TestCase):
    def test_dry_run_writes_state_report(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                result = runner.invoke(app, ["accelerate", "--dry-run", "--json"], catch_exceptions=False)
                self.assertEqual(result.exit_code, 0)
                self.assertIn("Accelerate Summary", result.output)
                self.assertTrue((Path(tmp) / ".hydra" / "state" / "accelerate_latest.json").exists())
            finally:
                os.chdir(previous)

    def test_help_shows_accelerate(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["accelerate", "--help"], catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Hydra Accelerate", result.output)


if __name__ == "__main__":
    unittest.main()
