from __future__ import annotations

import unittest

from continuum.doctor.models import CheckResult, EnvironmentInfo, Report, Status


class TestModels(unittest.TestCase):
    def test_check_result_rejects_invalid_severity(self) -> None:
        with self.assertRaises(ValueError):
            CheckResult(
                id="x",
                title="x",
                category="x",
                status=Status.PASS,
                message="x",
                severity=5,
            )

    def test_check_result_rejects_negative_duration(self) -> None:
        with self.assertRaises(ValueError):
            CheckResult(
                id="x",
                title="x",
                category="x",
                status=Status.PASS,
                message="x",
                duration_ms=-0.1,
            )

    def test_check_result_to_dict(self) -> None:
        result = CheckResult(
            id="environment.python_version",
            title="Python Version",
            category="environment",
            status=Status.PASS,
            message="ok",
            details={"current": "3.12.3"},
            remediation=["none"],
            severity=0,
            duration_ms=1.5,
        )
        payload = result.to_dict()

        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["details"], {"current": "3.12.3"})
        self.assertEqual(payload["remediation"], ["none"])

    def test_report_rejects_negative_total_duration(self) -> None:
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

        with self.assertRaises(ValueError):
            Report(
                schema_version="1.0.0",
                environment=env,
                total_duration_ms=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
