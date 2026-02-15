from __future__ import annotations

import os
import sys
from typing import Any

from continuum.doctor.checks.base import BaseCheck, Context, register_check
from continuum.doctor.models import CheckResult, Status
from continuum.doctor.utils.platform import is_container, is_wsl


@register_check
class PythonVersionCheck(BaseCheck):
    id = "environment.python_version"
    title = "Python Version"
    category = "environment"

    def run(self, context: Context) -> CheckResult:
        current = sys.version_info
        minimum = (3, 10)

        details: dict[str, Any] = {
            "current": f"{current.major}.{current.minor}.{current.micro}",
            "minimum": "3.10",
        }

        if (current.major, current.minor) < minimum:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.FAIL,
                message="Python 3.10+ is required.",
                details=details,
                remediation=[
                    "Install Python 3.10 or newer.",
                    "Create and activate a virtual environment with: python3.10 -m venv .venv",
                    "Activate it with: source .venv/bin/activate",
                ],
                severity=3,
            )

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.PASS,
            message="Python version is supported.",
            details=details,
            remediation=None,
            severity=0,
        )


@register_check
class VirtualEnvironmentCheck(BaseCheck):
    id = "environment.venv"
    title = "Virtual Environment"
    category = "environment"

    def run(self, context: Context) -> CheckResult:
        in_venv = (
            hasattr(sys, "real_prefix")
            or sys.prefix != getattr(sys, "base_prefix", sys.prefix)
            or bool(os.environ.get("CONDA_PREFIX"))
            or bool(os.environ.get("CONDA_DEFAULT_ENV"))
        )

        details = {
            "in_venv": in_venv,
            "sys_prefix": sys.prefix,
            "sys_base_prefix": getattr(sys, "base_prefix", sys.prefix),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
        }

        if in_venv:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.PASS,
                message="Python environment is isolated (venv/conda).",
                details=details,
                remediation=None,
                severity=0,
            )

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.WARN,
            message="Running on system Python; isolation is recommended.",
            details=details,
            remediation=[
                "Create a virtual environment: python3 -m venv .venv",
                "Activate it: source .venv/bin/activate",
            ],
            severity=1,
        )


@register_check
class RuntimeEnvironmentCheck(BaseCheck):
    id = "environment.runtime"
    title = "Runtime Environment"
    category = "environment"

    def run(self, context: Context) -> CheckResult:
        container_flag = is_container()
        wsl_flag = is_wsl()

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.PASS,
            message="Runtime environment detected.",
            details={
                "is_container": container_flag,
                "is_wsl": wsl_flag,
            },
            remediation=None,
            severity=0,
        )


__all__ = [
    "PythonVersionCheck",
    "VirtualEnvironmentCheck",
    "RuntimeEnvironmentCheck",
]
