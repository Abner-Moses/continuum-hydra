from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterable

from continuum.doctor.checks.base import BaseCheck, list_checks
from continuum.doctor.models import CheckResult, EnvironmentInfo, Report, Status
from continuum.doctor.utils.platform import (
    get_hostname,
    get_os_string,
    get_python_executable,
    get_python_version_string,
    is_container,
    is_wsl,
)


class DoctorRunner:
    def __init__(
        self,
        hydra_version: str,
        checks: Iterable[BaseCheck | type[BaseCheck]] | None = None,
        schema_version: str = "1.0.0",
    ) -> None:
        self.hydra_version = hydra_version
        self.schema_version = schema_version
        self.checks = self._resolve_checks(checks)

    @staticmethod
    def _resolve_checks(
        checks: Iterable[BaseCheck | type[BaseCheck]] | None,
    ) -> list[BaseCheck]:
        source = checks if checks is not None else list_checks()
        resolved: list[BaseCheck] = []

        for item in source:
            if isinstance(item, type):
                resolved.append(item())
            else:
                resolved.append(item)

        return resolved

    def build_environment(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            os=get_os_string(),
            python_version=get_python_version_string(),
            python_executable=get_python_executable(),
            is_container=is_container(),
            is_wsl=is_wsl(),
            hydra_version=self.hydra_version,
            hostname=get_hostname(),
        )

    def run(self, context: dict[str, Any] | None = None) -> Report:
        environment = self.build_environment()
        runtime_context: dict[str, Any] = dict(context or {})
        runtime_context.setdefault("environment", environment.to_dict())

        results: list[CheckResult] = []

        for check in self.checks:
            try:
                if hasattr(check, "should_run") and not check.should_run(runtime_context):
                    results.append(
                        CheckResult(
                            id=check.id,
                            title=check.title,
                            category=check.category,
                            status=Status.SKIP,
                            message="Check skipped",
                            severity=0,
                            duration_ms=0.0,
                        )
                    )
                    continue

                started = perf_counter()
                result = check.run(runtime_context)
                elapsed_ms = (perf_counter() - started) * 1000.0

                # Preserve reported duration when provided, else use measured runtime.
                duration_ms = result.duration_ms if result.duration_ms > 0 else elapsed_ms
                if duration_ms != result.duration_ms:
                    result = CheckResult(
                        id=result.id,
                        title=result.title,
                        category=result.category,
                        status=result.status,
                        message=result.message,
                        details=result.details,
                        remediation=result.remediation,
                        severity=result.severity,
                        duration_ms=duration_ms,
                    )

                results.append(result)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    CheckResult(
                        id=getattr(check, "id", check.__class__.__name__),
                        title=getattr(check, "title", check.__class__.__name__),
                        category=getattr(check, "category", "unknown"),
                        status=Status.ERROR,
                        message="Check raised an exception",
                        details={
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                        },
                        remediation=None,
                        severity=4,
                        duration_ms=0.0,
                    )
                )

        summary = self._compute_summary(results)
        overall_status = self._compute_overall_status(summary)
        total_duration_ms = sum(result.duration_ms for result in results)

        return Report(
            schema_version=self.schema_version,
            environment=environment,
            checks=results,
            summary=summary,
            overall_status=overall_status,
            total_duration_ms=total_duration_ms,
        )

    @staticmethod
    def _compute_summary(results: list[CheckResult]) -> dict[str, int]:
        counts = {status.value: 0 for status in Status}
        for result in results:
            counts[result.status.value] += 1
        return counts

    @staticmethod
    def _compute_overall_status(summary: dict[str, int]) -> str:
        if summary.get(Status.FAIL.value, 0) > 0:
            return "failed"
        if summary.get(Status.ERROR.value, 0) > 0:
            return "error"
        if summary.get(Status.WARN.value, 0) > 0:
            return "warnings"
        return "healthy"

    @staticmethod
    def exit_code(report: Report) -> int:
        if report.overall_status == "healthy":
            return 0
        if report.overall_status == "warnings":
            return 1
        if report.overall_status in {"failed", "error"}:
            return 2
        return 4


__all__ = ["DoctorRunner"]
