from __future__ import annotations

import json
import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

if find_spec("typer") is not None:
    from typer.testing import CliRunner

    from continuum.cli import app
else:
    CliRunner = None
    app = None


@unittest.skipIf(find_spec("typer") is None, "typer is not installed in this interpreter")
class TestAccelerateSystemCli(unittest.TestCase):
    def test_status_without_state(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                result = runner.invoke(app, ["accelerate", "--status"], catch_exceptions=False)
                self.assertEqual(result.exit_code, 0)
                self.assertIn("Active", result.stdout)
            finally:
                os.chdir(previous)

    def test_on_dry_run_does_not_write_state(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                result = runner.invoke(app, ["accelerate", "--on", "--dry-run"], catch_exceptions=False)
                self.assertEqual(result.exit_code, 0)
                self.assertFalse((Path(tmp) / ".hydra" / "state" / "accelerate_state.json").exists())
            finally:
                os.chdir(previous)

    def test_off_dry_run_with_state(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                state_path = Path(tmp) / ".hydra" / "state"
                state_path.mkdir(parents=True, exist_ok=True)
                payload = {
                    "active": True,
                    "platform": "linux",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "changes_applied": [],
                    "previous_state": {"cpu_governor": "powersave"},
                    "failures": [],
                    "applied_actions": ["cpu_governor"],
                }
                (state_path / "accelerate_state.json").write_text(json.dumps(payload), encoding="utf-8")

                result = runner.invoke(app, ["accelerate", "--off", "--dry-run"], catch_exceptions=False)
                self.assertEqual(result.exit_code, 0)
                # dry-run restore should not rewrite/clear the original payload on disk
                restored = json.loads((state_path / "accelerate_state.json").read_text(encoding="utf-8"))
                self.assertTrue(restored["active"])
            finally:
                os.chdir(previous)

    def test_partial_status_when_only_one_applied(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                with patch("continuum.accelerate.system_cli.detect_context", return_value={"platform": "linux"}):
                    with patch("continuum.accelerate.system_cli.capture_previous_state", return_value={}):
                        with patch(
                            "continuum.accelerate.system_cli.apply_acceleration",
                            return_value=(
                                [
                                    {
                                        "name": "cpu_governor",
                                        "result": "skipped",
                                        "message": "root required",
                                        "requires_root": True,
                                        "category": "cpu",
                                    },
                                    {
                                        "name": "ulimit_nofile",
                                        "result": "applied",
                                        "message": "soft limit set",
                                        "requires_root": False,
                                        "category": "cpu",
                                    },
                                ],
                                [],
                                ["ulimit_nofile"],
                                {"rlimit_nofile": {"soft": 1024, "hard": 4096}},
                            ),
                        ):
                            result = runner.invoke(app, ["accelerate", "--on"], catch_exceptions=False)
                self.assertEqual(result.exit_code, 0)

                state = json.loads((Path(tmp) / ".hydra" / "state" / "accelerate_state.json").read_text(encoding="utf-8"))
                self.assertEqual(state["active_status"], "Partial")
                self.assertEqual(state["applied_count"], 1)
                self.assertGreater(state["skipped_count"], 0)
            finally:
                os.chdir(previous)


class TestAccelerateSystemTuner(unittest.TestCase):
    def test_restore_does_not_attempt_swappiness_when_not_applied(self) -> None:
        from continuum.accelerate.system_tuner import restore_acceleration

        ctx = {"is_linux": True, "is_root": True, "nvidia_present": False, "is_windows": False}

        with patch("continuum.accelerate.system_tuner._run_cmd") as run_cmd:
            restore_acceleration(
                ctx,
                previous_state={"swappiness": 60, "rlimit_nofile": {"soft": 1024, "hard": 2048}},
                applied_actions=["ulimit_nofile"],
                dry_run=False,
            )

        commands = [call.args[0] for call in run_cmd.call_args_list]
        self.assertFalse(any(cmd[:2] == ["sysctl", "-w"] for cmd in commands))

    def test_change_records_are_deterministically_sorted(self) -> None:
        from continuum.accelerate.system_tuner import apply_acceleration

        ctx = {"is_linux": False, "is_windows": False, "is_macos": True, "is_root": False, "nvidia_present": False}
        changes, _failures, _applied, _state = apply_acceleration(
            ctx,
            previous_state={},
            dry_run=True,
            cpu_only=False,
            gpu_only=False,
            allow_risky=False,
        )
        names = [item["name"] for item in changes]
        self.assertEqual(names, sorted(names))


if __name__ == "__main__":
    unittest.main()
