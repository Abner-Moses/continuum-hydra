from __future__ import annotations

import os
import platform

from continuum.doctor.checks.base import BaseCheck, Context, register_check
from continuum.doctor.models import CheckResult, Status

_GIB = 1024**3


@register_check
class DevShmCheck(BaseCheck):
    id = "system.dev_shm"
    title = "/dev/shm Size"
    category = "system"

    def should_run(self, context: Context) -> bool:
        return platform.system() == "Linux"

    def run(self, context: Context) -> CheckResult:
        try:
            stat = os.statvfs("/dev/shm")
            total_bytes = int(stat.f_frsize * stat.f_blocks)
        except OSError as exc:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.FAIL,
                message="Unable to read /dev/shm capacity.",
                details={"error": f"{type(exc).__name__}: {exc}"},
                remediation=[
                    "Ensure /dev/shm is mounted and accessible.",
                    "In containers, set --shm-size=8g or mount tmpfs for /dev/shm.",
                ],
                severity=3,
            )

        total_gib = total_bytes / _GIB
        details = {
            "total_bytes": total_bytes,
            "total_gib": round(total_gib, 3),
        }

        if total_bytes < _GIB:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.FAIL,
                message="/dev/shm is below 1 GiB.",
                details=details,
                remediation=[
                    "Increase shared memory size (example: docker run --shm-size=8g ...).",
                    "Use tmpfs mount for /dev/shm when needed.",
                ],
                severity=3,
            )

        if total_bytes < 8 * _GIB:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.WARN,
                message="/dev/shm is below recommended 8 GiB.",
                details=details,
                remediation=[
                    "Increase shared memory size (example: docker run --shm-size=8g ...).",
                    "Use tmpfs mount for /dev/shm when needed.",
                ],
                severity=1,
            )

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.PASS,
            message="/dev/shm size is within recommended range.",
            details=details,
            remediation=None,
            severity=0,
        )


__all__ = ["DevShmCheck"]
