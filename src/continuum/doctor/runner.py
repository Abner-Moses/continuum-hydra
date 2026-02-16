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

    @staticmethod
    def filter_checks(
        checks: Iterable[BaseCheck],
        only: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> list[BaseCheck]:
        only_set = {item.strip() for item in (only or set()) if item.strip()}
        exclude_set = {item.strip() for item in (exclude or set()) if item.strip()}

        filtered: list[BaseCheck] = []
        for check in checks:
            if check.id in exclude_set:
                continue
            if only_set and check.id not in only_set and check.category not in only_set:
                continue
            filtered.append(check)
        return filtered

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
        runtime_context: dict[str, Any] = dict(context or {})
        deterministic = bool(runtime_context.get("deterministic", False))

        environment = self.build_environment()
        if deterministic:
            environment = EnvironmentInfo(
                timestamp_utc="1970-01-01T00:00:00Z",
                os=environment.os,
                python_version=environment.python_version,
                python_executable=environment.python_executable,
                is_container=environment.is_container,
                is_wsl=environment.is_wsl,
                hydra_version=environment.hydra_version,
                hostname=environment.hostname,
            )

        runtime_context.setdefault("environment", environment.to_dict())
        runtime_context.setdefault("os", environment.os)
        runtime_context.setdefault("is_container", environment.is_container)
        runtime_context.setdefault("is_wsl", environment.is_wsl)
        runtime_context.setdefault("results", {})
        runtime_context.setdefault("facts", {})

        if not isinstance(runtime_context["results"], dict):
            runtime_context["results"] = {}
        if not isinstance(runtime_context["facts"], dict):
            runtime_context["facts"] = {}

        facts: dict[str, Any] = runtime_context["facts"]
        facts.setdefault("os", environment.os)
        facts.setdefault("is_container", environment.is_container)
        facts.setdefault("is_wsl", environment.is_wsl)

        results: list[CheckResult] = []

        for check in self.checks:
            try:
                if hasattr(check, "should_run") and not check.should_run(runtime_context):
                    skip_result = CheckResult(
                        id=check.id,
                        title=check.title,
                        category=check.category,
                        status=Status.SKIP,
                        message="Check skipped",
                        severity=0,
                        duration_ms=0.0,
                    )
                    results.append(skip_result)
                    runtime_context["results"][skip_result.id] = skip_result
                    continue

                started = perf_counter() if not deterministic else 0.0
                result = check.run(runtime_context)
                elapsed_ms = 0.0 if deterministic else (perf_counter() - started) * 1000.0

                # Preserve reported duration when provided, else use measured runtime.
                duration_ms = 0.0 if deterministic else (result.duration_ms if result.duration_ms > 0 else elapsed_ms)
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
                runtime_context["results"][result.id] = result
            except Exception as exc:  # noqa: BLE001
                error_result = CheckResult(
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
                results.append(error_result)
                runtime_context["results"][error_result.id] = error_result

        summary = self._compute_summary(results)
        overall_status = self._compute_overall_status(summary)
        total_duration_ms = 0.0 if deterministic else sum(result.duration_ms for result in results)

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
