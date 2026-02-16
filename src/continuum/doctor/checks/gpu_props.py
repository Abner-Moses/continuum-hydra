from __future__ import annotations

from continuum.doctor.checks.base import BaseCheck, Context, register_check
from continuum.doctor.models import CheckResult, Status


def _results(context: Context) -> dict[str, CheckResult]:
    payload = context.get("results")
    return payload if isinstance(payload, dict) else {}


@register_check
class GpuDevicePropertiesCheck(BaseCheck):
    id = "gpu.device_properties"
    title = "GPU Device Properties"
    category = "gpu"

    def should_run(self, context: Context) -> bool:
        installed = _results(context).get("pytorch.installed")
        return installed is not None and installed.status == Status.PASS

    def run(self, context: Context) -> CheckResult:
        try:
            import torch  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.FAIL,
                message="PyTorch import failed while collecting GPU properties.",
                details={"import_error": f"{type(exc).__name__}: {exc}"},
                remediation=[
                    "Reinstall PyTorch in the active environment.",
                ],
                severity=3,
            )

        if not bool(torch.cuda.is_available()):
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.SKIP,
                message="torch.cuda is not available; skipping GPU property collection.",
                details={"cuda_available": False},
                remediation=None,
                severity=0,
                duration_ms=0.0,
            )

        device_count = int(torch.cuda.device_count())
        device_props: list[dict[str, object]] = []
        low_cc: list[dict[str, object]] = []

        for idx in range(min(device_count, 8)):
            props = torch.cuda.get_device_properties(idx)
            major = int(getattr(props, "major", 0))
            minor = int(getattr(props, "minor", 0))
            cc = float(f"{major}.{minor}")
            item = {
                "index": idx,
                "name": str(getattr(props, "name", f"GPU-{idx}")),
                "compute_capability": cc,
                "total_memory": int(getattr(props, "total_memory", 0)),
                "multiprocessor_count": int(getattr(props, "multi_processor_count", 0)),
            }
            device_props.append(item)
            if cc < 7.0:
                low_cc.append(item)

        details = {"device_count": device_count, "devices": device_props}
        if low_cc:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.WARN,
                message="Detected GPU(s) with compute capability below 7.0.",
                details=details,
                remediation=[
                    "Mixed precision and tensor-core acceleration may be limited on older GPUs.",
                    "Verify capabilities with: python -c \"import torch; print([torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])\"",
                ],
                severity=1,
            )

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.PASS,
            message="Collected GPU device properties.",
            details=details,
            remediation=None,
            severity=0,
        )


__all__ = ["GpuDevicePropertiesCheck"]
