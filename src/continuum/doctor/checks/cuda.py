from __future__ import annotations

import platform
import re
import shutil
import subprocess
from pathlib import Path

try:
    from packaging.version import Version
except Exception:  # pragma: no cover
    class Version:  # type: ignore[no-redef]
        def __init__(self, value: str) -> None:
            self.parts = tuple(int(part) for part in re.findall(r"\d+", value))

        def __ge__(self, other: object) -> bool:
            if not isinstance(other, Version):
                return NotImplemented
            max_len = max(len(self.parts), len(other.parts))
            left = self.parts + (0,) * (max_len - len(self.parts))
            right = other.parts + (0,) * (max_len - len(other.parts))
            return left >= right

from continuum.doctor.checks.base import BaseCheck, Context, register_check
from continuum.doctor.models import CheckResult, Status

_MAX_CAPTURE_LEN = 2000
_CUDA_DRIVER_MIN = {
    "11.8": "520.61.05",
    "12.1": "530.30.02",
    "12.2": "535.54.03",
    "12.3": "545.23.06",
    "12.4": "550.54.14",
}


def _truncate_text(value: str | None, limit: int = _MAX_CAPTURE_LEN) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated>"


def _is_linux_or_windows() -> bool:
    return platform.system() in {"Linux", "Windows"}


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


def _extract_version(text: str) -> str | None:
    match = re.search(r"(\d+\.\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def _driver_version_from_nvml() -> str | None:
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        raw = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace").strip()
        return str(raw).strip()
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def _driver_version_from_nvidia_smi() -> tuple[str | None, dict[str, object]]:
    smi_path = shutil.which("nvidia-smi")
    details: dict[str, object] = {
        "nvidia_smi_path": smi_path,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "parse_source": None,
    }
    if smi_path is None:
        return None, details

    try:
        query = subprocess.run(
            [smi_path, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        details["returncode"] = query.returncode
        details["stdout"] = _truncate_text(query.stdout)
        details["stderr"] = _truncate_text(query.stderr)
        if query.returncode == 0:
            first = next((line.strip() for line in query.stdout.splitlines() if line.strip()), "")
            version = _extract_version(first)
            if version is not None:
                details["parse_source"] = "query-gpu"
                return version, details

        fallback = subprocess.run(
            [smi_path],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        details["returncode"] = fallback.returncode
        details["stdout"] = _truncate_text(fallback.stdout)
        details["stderr"] = _truncate_text(fallback.stderr)
        if fallback.returncode == 0:
            line_match = re.search(r"Driver Version:\s*([0-9][0-9.\-]*)", fallback.stdout)
            if line_match:
                details["parse_source"] = "default-output"
                return line_match.group(1), details
    except Exception as exc:  # noqa: BLE001
        details["stderr"] = _truncate_text(str(exc))

    return None, details


def _get_cuda_version_from_facts(context: Context) -> str | None:
    facts = _facts(context)
    for key in ("torch_cuda_version", "nvcc_version"):
        value = facts.get(key)
        if isinstance(value, str) and value:
            return value
    return None


@register_check
class CudaDriverVersionCheck(BaseCheck):
    id = "cuda.driver_version"
    title = "CUDA Driver Version"
    category = "driver"

    def should_run(self, context: Context) -> bool:
        return _is_linux_or_windows()

    def run(self, context: Context) -> CheckResult:
        facts = _facts(context)
        nvml_version = _driver_version_from_nvml()
        if nvml_version:
            facts["driver_version"] = nvml_version
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.PASS,
                message="Detected NVIDIA driver version via NVML.",
                details={"driver_version": nvml_version, "method_used": "nvml"},
                remediation=None,
                severity=0,
            )

        smi_version, smi_details = _driver_version_from_nvidia_smi()
        smi_present = bool(smi_details.get("nvidia_smi_path"))
        if smi_version:
            facts["driver_version"] = smi_version
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.PASS,
                message="Detected NVIDIA driver version via nvidia-smi.",
                details={
                    "driver_version": smi_version,
                    "method_used": "nvidia-smi",
                    **smi_details,
                },
                remediation=None,
                severity=0,
            )

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.FAIL if smi_present else Status.WARN,
            message="Unable to detect NVIDIA driver version.",
            details={
                "driver_version": None,
                "method_used": "unknown",
                **smi_details,
            },
            remediation=[
                "Install or repair NVIDIA drivers.",
                "Run nvidia-smi to verify driver visibility.",
                "Verify NVML access with: python -c \"import pynvml; pynvml.nvmlInit(); print('ok')\"",
            ],
            severity=3 if smi_present else 1,
        )


@register_check
class CudaToolkitNvccCheck(BaseCheck):
    id = "cuda.toolkit_nvcc"
    title = "CUDA Toolkit (nvcc)"
    category = "cuda"

    def should_run(self, context: Context) -> bool:
        return _is_linux_or_windows()

    def run(self, context: Context) -> CheckResult:
        facts = _facts(context)
        nvcc_path = shutil.which("nvcc")
        details = {
            "nvcc_found": nvcc_path is not None,
            "nvcc_path": nvcc_path,
            "nvcc_version": None,
            "stdout": "",
            "stderr": "",
            "returncode": None,
        }

        if nvcc_path is None:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.WARN,
                message="nvcc not found; runtime-only CUDA environments are common.",
                details=details,
                remediation=[
                    "Install CUDA toolkit if nvcc/toolchain workflows are required.",
                ],
                severity=1,
            )

        try:
            proc = subprocess.run(
                [nvcc_path, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            details["returncode"] = proc.returncode
            details["stdout"] = _truncate_text(proc.stdout)
            details["stderr"] = _truncate_text(proc.stderr)
            version = None
            rel_match = re.search(r"release\s+(\d+\.\d+)", proc.stdout, flags=re.IGNORECASE)
            if rel_match:
                version = rel_match.group(1)
            if version is None:
                version = _extract_version(proc.stdout)
            details["nvcc_version"] = version
            if proc.returncode == 0 and version is not None:
                facts["nvcc_version"] = version
                return CheckResult(
                    id=self.id,
                    title=self.title,
                    category=self.category,
                    status=Status.PASS,
                    message="nvcc detected with parseable CUDA toolkit version.",
                    details=details,
                    remediation=None,
                    severity=0,
                )
        except Exception as exc:  # noqa: BLE001
            details["stderr"] = _truncate_text(str(exc))

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.WARN,
            message="nvcc found but CUDA toolkit version could not be parsed.",
            details=details,
            remediation=[
                "Verify CUDA toolkit installation if compile-time tooling is needed.",
            ],
            severity=1,
        )


@register_check
class CudaTorchCudaVersionCheck(BaseCheck):
    id = "cuda.torch_cuda_version"
    title = "Torch CUDA Version"
    category = "cuda"

    def should_run(self, context: Context) -> bool:
        installed = _results(context).get("pytorch.installed")
        return installed is not None and installed.status == Status.PASS

    def run(self, context: Context) -> CheckResult:
        facts = _facts(context)
        try:
            import torch  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.FAIL,
                message="Unable to import torch for CUDA version detection.",
                details={"import_error": f"{type(exc).__name__}: {exc}"},
                remediation=[
                    "Reinstall PyTorch in the active environment.",
                ],
                severity=3,
            )

        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        facts["torch_cuda_version"] = cuda_version
        gpu_count = facts.get("gpu_count", 0)
        gpu_present = isinstance(gpu_count, int) and gpu_count > 0

        details = {
            "torch_version": str(getattr(torch, "__version__", "unknown")),
            "torch_cuda_version": cuda_version,
            "gpu_count": gpu_count,
        }

        if cuda_version is None and gpu_present:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.WARN,
                message="GPU detected but torch.version.cuda is not set.",
                details=details,
                remediation=[
                    "Install a CUDA-enabled PyTorch build for this environment.",
                ],
                severity=1,
            )

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.PASS,
            message="Collected torch CUDA version metadata.",
            details=details,
            remediation=None,
            severity=0,
        )


@register_check
class CudaDriverCompatCheck(BaseCheck):
    id = "cuda.driver_cuda_compat"
    title = "CUDA Driver Compatibility"
    category = "integration"

    def should_run(self, context: Context) -> bool:
        facts = _facts(context)
        driver_version = facts.get("driver_version")
        cuda_version = _get_cuda_version_from_facts(context)
        return isinstance(driver_version, str) and bool(driver_version) and isinstance(cuda_version, str) and bool(cuda_version)

    def run(self, context: Context) -> CheckResult:
        facts = _facts(context)
        driver_version = str(facts.get("driver_version"))
        cuda_version_raw = str(_get_cuda_version_from_facts(context))
        cuda_key_match = re.match(r"^(\d+\.\d+)", cuda_version_raw)
        cuda_key = cuda_key_match.group(1) if cuda_key_match else cuda_version_raw
        required = _CUDA_DRIVER_MIN.get(cuda_key)

        details = {
            "detected_cuda_version": cuda_version_raw,
            "driver_version": driver_version,
            "required_min_driver": required,
        }

        if required is None:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.WARN,
                message="CUDA version not present in built-in compatibility table.",
                details=details,
                remediation=[
                    "Verify NVIDIA's CUDA compatibility matrix for this CUDA release.",
                    "Reference: https://docs.nvidia.com/deploy/cuda-compatibility/",
                ],
                severity=1,
            )

        try:
            driver_ok = Version(driver_version) >= Version(required)
        except Exception:  # noqa: BLE001
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.WARN,
                message="Could not compare driver and CUDA versions.",
                details=details,
                remediation=[
                    f"Ensure NVIDIA driver is >= {required} for CUDA {cuda_key}.",
                ],
                severity=1,
            )

        if not driver_ok:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.FAIL,
                message="NVIDIA driver is below the minimum required for detected CUDA.",
                details=details,
                remediation=[
                    f"Upgrade NVIDIA driver to >= {required}.",
                ],
                severity=3,
            )

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.PASS,
            message="Driver/CUDA compatibility check passed against built-in matrix.",
            details=details,
            remediation=None,
            severity=0,
        )


@register_check
class CudaRuntimeHintCheck(BaseCheck):
    id = "cuda.runtime_hint"
    title = "CUDA Runtime Hint"
    category = "cuda"

    def should_run(self, context: Context) -> bool:
        return platform.system() == "Linux"

    def run(self, context: Context) -> CheckResult:
        cuda_root = Path("/usr/local/cuda")
        details = {
            "cuda_root_path": str(cuda_root),
            "cuda_root_exists": cuda_root.exists(),
        }
        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.PASS,
            message="Collected CUDA runtime path hint.",
            details=details,
            remediation=None,
            severity=0,
        )


__all__ = [
    "CudaDriverVersionCheck",
    "CudaToolkitNvccCheck",
    "CudaTorchCudaVersionCheck",
    "CudaDriverCompatCheck",
    "CudaRuntimeHintCheck",
]
