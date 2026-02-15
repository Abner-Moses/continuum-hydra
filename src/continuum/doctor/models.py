from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class CheckResult:
    id: str
    title: str
    category: str
    status: Status
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    remediation: list[str] | None = None
    severity: int = 0
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.severity <= 4:
            raise ValueError("severity must be between 0 and 4")
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "status": self.status.value,
            "message": self.message,
            "details": dict(self.details),
            "remediation": list(self.remediation) if self.remediation is not None else None,
            "severity": self.severity,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    timestamp_utc: str
    os: str
    python_version: str
    python_executable: str
    is_container: bool
    is_wsl: bool
    hydra_version: str
    hostname: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "os": self.os,
            "python_version": self.python_version,
            "python_executable": self.python_executable,
            "is_container": self.is_container,
            "is_wsl": self.is_wsl,
            "hydra_version": self.hydra_version,
            "hostname": self.hostname,
        }


@dataclass(frozen=True, slots=True)
class Report:
    schema_version: str
    environment: EnvironmentInfo
    checks: list[CheckResult] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)
    overall_status: Literal["healthy", "warnings", "failed", "error"] = "healthy"
    total_duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.total_duration_ms < 0:
            raise ValueError("total_duration_ms must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "environment": self.environment.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
            "summary": dict(self.summary),
            "overall_status": self.overall_status,
            "total_duration_ms": self.total_duration_ms,
        }


__all__ = [
    "Status",
    "CheckResult",
    "EnvironmentInfo",
    "Report",
]
