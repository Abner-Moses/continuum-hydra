from __future__ import annotations

import importlib.util

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


def _gpu_count(context: Context) -> int:
    facts = _facts(context)
    value = facts.get("gpu_count")
    return value if isinstance(value, int) else 0


@register_check
class PytorchInstalledCheck(BaseCheck):
    id = "pytorch.installed"
    title = "PyTorch Installation"
    category = "pytorch"

    def run(self, context: Context) -> CheckResult:
        spec = importlib.util.find_spec("torch")
        found = spec is not None
        origin = getattr(spec, "origin", None) if found else None

        _facts(context)["torch_installed"] = found

        if found:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.PASS,
                message="PyTorch package detected.",
                details={"found": found, "origin": origin},
                remediation=None,
                severity=0,
            )

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.FAIL,
            message="PyTorch package is not installed.",
            details={"found": found, "origin": origin},
            remediation=[
                "Install PyTorch for your platform: pip install torch",
            ],
            severity=3,
        )


@register_check
class PytorchCudaAvailableCheck(BaseCheck):
    id = "pytorch.cuda_available"
    title = "PyTorch CUDA Availability"
    category = "pytorch"

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
                message="PyTorch import failed.",
                details={"import_error": f"{type(exc).__name__}: {exc}"},
                remediation=[
                    "Reinstall PyTorch in the active environment.",
                    "Verify python/venv activation and package compatibility.",
                ],
                severity=3,
            )

        torch_version = str(getattr(torch, "__version__", "unknown"))
        torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        cuda_available = bool(torch.cuda.is_available())
        gpu_count = _gpu_count(context)
        gpu_present = gpu_count > 0
        cpu_build = (torch_cuda_version is None) or ("+cpu" in torch_version.lower())

        _facts(context)["torch_cuda_available"] = cuda_available
        details = {
            "torch_version": torch_version,
            "torch_cuda_version": torch_cuda_version,
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
        }

        if cuda_available:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.PASS,
                message="torch.cuda.is_available() is True.",
                details=details,
                remediation=None,
                severity=0,
            )

        if gpu_present:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.FAIL,
                message="GPU detected but torch.cuda.is_available() is False.",
                details=details,
                remediation=[
                    "Install a CUDA-enabled PyTorch build matching your driver/CUDA runtime.",
                    "Verify CUDA_VISIBLE_DEVICES is not masking GPUs.",
                ],
                severity=3,
            )

        if cpu_build:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.WARN,
                message="PyTorch appears to be CPU-only; CUDA is unavailable.",
                details=details,
                remediation=[
                    "Install a CUDA-enabled PyTorch build if GPU acceleration is required.",
                ],
                severity=1,
            )

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.WARN,
            message="CUDA is unavailable in the current PyTorch runtime.",
            details=details,
            remediation=[
                "Verify driver, CUDA runtime, and PyTorch build compatibility.",
            ],
            severity=1,
        )


@register_check
class PytorchCudaVersionCheck(BaseCheck):
    id = "pytorch.cuda_version"
    title = "PyTorch CUDA Version"
    category = "pytorch"

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
                message="PyTorch import failed while querying CUDA version.",
                details={"import_error": f"{type(exc).__name__}: {exc}"},
                remediation=[
                    "Reinstall PyTorch in the active environment.",
                ],
                severity=3,
            )

        torch_version = str(getattr(torch, "__version__", "unknown"))
        torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)

        cudnn_version: int | None = None
        try:
            cudnn_api = getattr(getattr(torch, "backends", None), "cudnn", None)
            if cudnn_api is not None and hasattr(cudnn_api, "version"):
                cudnn_version = cudnn_api.version()
        except Exception:  # noqa: BLE001
            cudnn_version = None

        gpu_count = _gpu_count(context)
        details = {
            "torch_version": torch_version,
            "torch_cuda_version": torch_cuda_version,
            "cudnn_version": cudnn_version,
            "gpu_count": gpu_count,
        }

        if torch_cuda_version is None and gpu_count > 0:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.WARN,
                message="GPU detected but PyTorch reports no CUDA version.",
                details=details,
                remediation=[
                    "Install a CUDA-enabled PyTorch build matching your platform.",
                ],
                severity=1,
            )

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.PASS,
            message="PyTorch CUDA version metadata collected.",
            details=details,
            remediation=None,
            severity=0,
        )


__all__ = [
    "PytorchInstalledCheck",
    "PytorchCudaAvailableCheck",
    "PytorchCudaVersionCheck",
]
