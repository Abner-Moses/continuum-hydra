from __future__ import annotations

import os
import platform

from continuum.doctor.checks.base import BaseCheck, Context, register_check
from continuum.doctor.models import CheckResult, Status


def _results(context: Context) -> dict[str, CheckResult]:
    payload = context.get("results")
    return payload if isinstance(payload, dict) else {}


def _facts(context: Context) -> dict[str, object]:
    payload = context.get("facts")
    if isinstance(payload, dict):
        return payload
    facts: dict[str, object] = {}
    context["facts"] = facts
    return facts


@register_check
class NcclEnvConfigCheck(BaseCheck):
    id = "nccl.env_config"
    title = "NCCL Environment Configuration"
    category = "nccl"

    def should_run(self, context: Context) -> bool:
        if platform.system() != "Linux":
            return False
        facts = _facts(context)
        gpu_count = facts.get("gpu_count")
        multi_gpu = isinstance(gpu_count, int) and gpu_count > 1
        return multi_gpu or bool(context.get("is_container"))

    def run(self, context: Context) -> CheckResult:
        facts = _facts(context)
        gpu_count = facts.get("gpu_count", 0)
        p2p_disable = os.environ.get("NCCL_P2P_DISABLE")
        ib_disable = os.environ.get("NCCL_IB_DISABLE")
        socket_ifname = os.environ.get("NCCL_SOCKET_IFNAME")

        suspicious: list[str] = []
        if isinstance(gpu_count, int) and gpu_count > 1 and str(p2p_disable).lower() in {"1", "true", "yes"}:
            suspicious.append("NCCL_P2P_DISABLE disables direct GPU P2P on multi-GPU setup.")
        if isinstance(gpu_count, int) and gpu_count > 1 and str(ib_disable).lower() in {"1", "true", "yes"}:
            suspicious.append("NCCL_IB_DISABLE disables InfiniBand transport.")
        if socket_ifname in {"lo", "docker0"}:
            suspicious.append("NCCL_SOCKET_IFNAME is set to loopback/docker interface.")

        details = {
            "gpu_count": gpu_count,
            "is_container": bool(context.get("is_container")),
            "NCCL_P2P_DISABLE": p2p_disable,
            "NCCL_IB_DISABLE": ib_disable,
            "NCCL_SOCKET_IFNAME": socket_ifname,
            "suspicious": suspicious,
        }

        if suspicious:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.WARN,
                message="Potentially problematic NCCL environment settings detected.",
                details=details,
                remediation=[
                    "Unset overrides and retry: unset NCCL_P2P_DISABLE NCCL_IB_DISABLE NCCL_SOCKET_IFNAME",
                    "Unset restrictive NCCL_* variables unless intentionally required.",
                    "Prefer explicit interface selection for production networks.",
                ],
                severity=1,
            )

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.PASS,
            message="No obvious NCCL environment misconfiguration detected.",
            details=details,
            remediation=None,
            severity=0,
        )


@register_check
class NcclTorchBackendCheck(BaseCheck):
    id = "nccl.torch_backend"
    title = "Torch NCCL Backend Availability"
    category = "nccl"

    def should_run(self, context: Context) -> bool:
        installed = _results(context).get("pytorch.installed")
        return installed is not None and installed.status == Status.PASS

    def run(self, context: Context) -> CheckResult:
        if platform.system() != "Linux":
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.SKIP,
                message="NCCL backend checks are Linux-only.",
                details={"platform": platform.system()},
                remediation=None,
                severity=0,
                duration_ms=0.0,
            )

        facts = _facts(context)
        gpu_count = facts.get("gpu_count", 0)
        multi_gpu = isinstance(gpu_count, int) and gpu_count > 1

        dist_available = False
        nccl_available = False
        import_error = None
        try:
            import torch.distributed as dist  # type: ignore[import-not-found]

            dist_available = bool(dist.is_available())
            if hasattr(dist, "is_nccl_available"):
                nccl_available = bool(dist.is_nccl_available())
        except Exception as exc:  # noqa: BLE001
            import_error = f"{type(exc).__name__}: {exc}"

        details = {
            "torch_distributed_available": dist_available,
            "nccl_available_boolean": nccl_available,
            "gpu_count": gpu_count,
            "import_error": import_error,
        }

        if multi_gpu and not nccl_available:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.WARN,
                message="Multi-GPU environment detected but NCCL backend is unavailable.",
                details=details,
                remediation=[
                    "Install a PyTorch build with distributed/NCCL support.",
                    "Verify NCCL runtime libraries are available on the system.",
                ],
                severity=1,
            )

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.PASS,
            message="Torch distributed/NCCL backend check completed.",
            details=details,
            remediation=None,
            severity=0,
        )


__all__ = [
    "NcclEnvConfigCheck",
    "NcclTorchBackendCheck",
]
