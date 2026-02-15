from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from continuum.doctor.formatters.json import write_report_json
from continuum.doctor.models import CheckResult, EnvironmentInfo, Report, Status


class TestJsonFormatter(unittest.TestCase):
    def test_write_report_json_creates_file_with_expected_shape(self) -> None:
        env = EnvironmentInfo(
            timestamp_utc="2026-01-01T00:00:00+00:00",
            os="Linux 6.8",
            python_version="3.12.3",
            python_executable="/usr/bin/python3",
            is_container=False,
            is_wsl=False,
            hydra_version="0.1.0",
            hostname="localhost",
        )
        report = Report(
            schema_version="1.0.0",
            environment=env,
            checks=[
                CheckResult(
                    id="test.pass",
                    title="Pass",
                    category="test",
                    status=Status.PASS,
                    message="ok",
                )
            ],
            summary={"PASS": 1, "WARN": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0},
            overall_status="healthy",
            total_duration_ms=1.0,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = write_report_json(report, Path(tmp))
            self.assertTrue(output.exists())

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["overall_status"], "healthy")
            self.assertEqual(payload["checks"][0]["id"], "test.pass")


if __name__ == "__main__":
    unittest.main()
