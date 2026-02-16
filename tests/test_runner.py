from __future__ import annotations

import unittest

from continuum.doctor.checks.base import BaseCheck
from continuum.doctor.models import CheckResult, Status
from continuum.doctor.runner import DoctorRunner


class PassingCheck(BaseCheck):
    id = "test.pass"
    title = "Pass"
    category = "test"

    def run(self, context):
        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.PASS,
            message="ok",
            duration_ms=0.25,
        )


class WarningCheck(BaseCheck):
    id = "test.warn"
    title = "Warn"
    category = "test"

    def run(self, context):
        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.WARN,
            message="warn",
            severity=1,
        )


class ExplodingCheck(BaseCheck):
    id = "test.explode"
    title = "Explode"
    category = "test"

    def run(self, context):
        raise RuntimeError("boom")


class SkippedCheck(BaseCheck):
    id = "test.skip"
    title = "Skip"
    category = "test"

    def should_run(self, context):
        return False

    def run(self, context):
        raise AssertionError("should not run")


class TimedCheck(BaseCheck):
    id = "timed.pass"
    title = "Timed"
    category = "timed"

    def run(self, context):
        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.PASS,
            message="timed",
            duration_ms=5.0,
        )


class TestDoctorRunner(unittest.TestCase):
    def test_run_handles_pass_warn_skip_error(self) -> None:
        runner = DoctorRunner(
            hydra_version="0.1.0",
            checks=[PassingCheck, WarningCheck, SkippedCheck, ExplodingCheck],
        )

        report = runner.run(context={"verbose": True})

        self.assertEqual(report.summary["PASS"], 1)
        self.assertEqual(report.summary["WARN"], 1)
        self.assertEqual(report.summary["SKIP"], 1)
        self.assertEqual(report.summary["ERROR"], 1)
        self.assertEqual(report.overall_status, "error")
        self.assertEqual(DoctorRunner.exit_code(report), 2)

    def test_exit_code_mappings(self) -> None:
        self.assertEqual(DoctorRunner.exit_code(_report_with_status("healthy")), 0)
        self.assertEqual(DoctorRunner.exit_code(_report_with_status("warnings")), 1)
        self.assertEqual(DoctorRunner.exit_code(_report_with_status("failed")), 2)
        self.assertEqual(DoctorRunner.exit_code(_report_with_status("error")), 2)

    def test_filter_checks_by_category_and_id_and_exclude(self) -> None:
        checks = DoctorRunner(hydra_version="0.1.0", checks=[PassingCheck, WarningCheck, TimedCheck]).checks
        filtered = DoctorRunner.filter_checks(checks, only={"test", "timed.pass"}, exclude={"test.warn"})
        ids = [check.id for check in filtered]
        self.assertEqual(ids, ["test.pass", "timed.pass"])

    def test_deterministic_mode_stabilizes_timestamp_and_durations(self) -> None:
        runner = DoctorRunner(hydra_version="0.1.0", checks=[TimedCheck])
        report = runner.run(context={"deterministic": True})

        self.assertEqual(report.environment.timestamp_utc, "1970-01-01T00:00:00Z")
        self.assertEqual(report.checks[0].duration_ms, 0.0)
        self.assertEqual(report.total_duration_ms, 0.0)


def _report_with_status(overall_status):
    runner = DoctorRunner(hydra_version="0.1.0", checks=[PassingCheck])
    report = runner.run()
    return type(report)(
        schema_version=report.schema_version,
        environment=report.environment,
        checks=report.checks,
        summary=report.summary,
        overall_status=overall_status,
        total_duration_ms=report.total_duration_ms,
    )


if __name__ == "__main__":
    unittest.main()
