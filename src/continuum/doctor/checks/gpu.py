from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

from continuum.doctor.checks.base import BaseCheck, Context, register_check
from continuum.doctor.models import CheckResult, Status

_MAX_CAPTURE_LEN = 2000


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


@register_check
class NvidiaSmiCheck(BaseCheck):
    id = "driver.nvidia_smi"
    title = "NVIDIA Driver CLI"
    category = "driver"

    def should_run(self, context: Context) -> bool:
        return _is_linux_or_windows()

    def run(self, context: Context) -> CheckResult:
        smi_path = shutil.which("nvidia-smi")
        details = {
            "detection_method": "PATH lookup + nvidia-smi -L",
            "binary_path": smi_path,
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }

        facts = _facts(context)
        facts["nvidia_smi_ok"] = False

        if smi_path is None:
            details["stderr"] = "nvidia-smi not found in PATH."
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.FAIL,
                message="nvidia-smi command not found.",
                details=details,
                remediation=[
                    "Install NVIDIA drivers for your platform.",
                    "Verify nvidia-smi is in PATH.",
                ],
                severity=3,
            )

        try:
            proc = subprocess.run(
                [smi_path, "-L"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            details["returncode"] = proc.returncode
            details["stdout"] = _truncate_text(proc.stdout)
            details["stderr"] = _truncate_text(proc.stderr)
        except Exception as exc:  # noqa: BLE001
            details["stderr"] = _truncate_text(str(exc))
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.FAIL,
                message="Failed to execute nvidia-smi.",
                details=details,
                remediation=[
                    "Install NVIDIA drivers for your platform.",
                    "Verify nvidia-smi is in PATH and executable.",
                ],
                severity=3,
            )

        if proc.returncode == 0:
            facts["nvidia_smi_ok"] = True
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.PASS,
                message="nvidia-smi detected and operational.",
                details=details,
                remediation=None,
                severity=0,
            )

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.FAIL,
            message="nvidia-smi returned a non-zero exit code.",
            details=details,
            remediation=[
                "Install NVIDIA drivers for your platform.",
                "Verify nvidia-smi is in PATH and functional.",
            ],
            severity=3,
        )


@register_check
class NvmlAvailableCheck(BaseCheck):
    id = "gpu.nvml_available"
    title = "NVML Availability"
    category = "gpu"

    def should_run(self, context: Context) -> bool:
        return _is_linux_or_windows()

    def run(self, context: Context) -> CheckResult:
        facts = _facts(context)
        facts["nvml_ok"] = False

        import_error: str | None = None
        nvml_error: str | None = None

        try:
            import pynvml  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            import_error = f"{type(exc).__name__}: {exc}"
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.WARN,
                message="pynvml is not available; NVML checks limited.",
                details={
                    "import_error": import_error,
                    "nvml_error": nvml_error,
                },
                remediation=[
                    "Install NVML bindings: pip install pynvml",
                ],
                severity=1,
            )

        try:
            pynvml.nvmlInit()
            facts["nvml_ok"] = True
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.PASS,
                message="NVML initialized successfully.",
                details={
                    "import_error": import_error,
                    "nvml_error": nvml_error,
                },
                remediation=None,
                severity=0,
            )
        except Exception as exc:  # noqa: BLE001
            nvml_error = f"{type(exc).__name__}: {exc}"
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.FAIL,
                message="NVML initialization failed.",
                details={
                    "import_error": import_error,
                    "nvml_error": nvml_error,
                },
                remediation=[
                    "Install/repair NVIDIA drivers and verify NVML is available.",
                    "Install NVML bindings: pip install pynvml",
                ],
                severity=3,
            )
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass


@register_check
class NvmlDevicesCheck(BaseCheck):
    id = "gpu.nvml_devices"
    title = "NVML Device Enumeration"
    category = "gpu"

    def should_run(self, context: Context) -> bool:
        if not _is_linux_or_windows():
            return False
        nvml_result = _results(context).get("gpu.nvml_available")
        return nvml_result is not None and nvml_result.status == Status.PASS

    def run(self, context: Context) -> CheckResult:
        facts = _facts(context)
        details = {
            "gpu_count": 0,
            "gpu_names": [],
        }

        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            count = int(pynvml.nvmlDeviceGetCount())
            names: list[str] = []

            for idx in range(min(count, 8)):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                raw_name = pynvml.nvmlDeviceGetName(handle)
                names.append(raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name))

            details["gpu_count"] = count
            details["gpu_names"] = names

            facts["gpu_count"] = count
            facts["gpu_names"] = names

            if count > 0:
                return CheckResult(
                    id=self.id,
                    title=self.title,
                    category=self.category,
                    status=Status.PASS,
                    message=f"Detected {count} GPU device(s) via NVML.",
                    details=details,
                    remediation=None,
                    severity=0,
                )

            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.FAIL,
                message="NVML initialized but no GPU devices were found.",
                details=details,
                remediation=[
                    "Verify NVIDIA driver installation.",
                    "If running in container, enable GPU passthrough: docker run --gpus all ...",
                    "Check CUDA_VISIBLE_DEVICES and runtime constraints.",
                ],
                severity=3,
            )
        except Exception as exc:  # noqa: BLE001
            details["nvml_error"] = f"{type(exc).__name__}: {exc}"
            facts["gpu_count"] = 0
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.FAIL,
                message="Failed to enumerate GPU devices with NVML.",
                details=details,
                remediation=[
                    "Verify NVIDIA driver installation.",
                    "If running in container, enable GPU passthrough: docker run --gpus all ...",
                    "Check CUDA_VISIBLE_DEVICES and runtime constraints.",
                ],
                severity=3,
            )
        finally:
            try:
                import pynvml  # type: ignore[import-not-found]

                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass


@register_check
class RuntimeGpuPassthroughCheck(BaseCheck):
    id = "runtime.gpu_passthrough"
    title = "Container GPU Passthrough"
    category = "integration"

    def should_run(self, context: Context) -> bool:
        return platform.system() == "Linux" and bool(context.get("is_container"))

    def run(self, context: Context) -> CheckResult:
        device_nodes = sorted(str(path) for path in Path("/dev").glob("nvidia*"))

        smi_path = shutil.which("nvidia-smi")
        returncode: int | None = None
        stderr = ""
        if smi_path is not None:
            try:
                proc = subprocess.run(
                    [smi_path, "-L"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                returncode = proc.returncode
                stderr = _truncate_text(proc.stderr)
            except Exception as exc:  # noqa: BLE001
                stderr = _truncate_text(str(exc))

        visible = bool(device_nodes) or returncode == 0
        details = {
            "device_nodes": device_nodes[:16],
            "nvidia_smi_returncode": returncode,
            "nvidia_smi_stderr": stderr,
        }

        if visible:
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.PASS,
                message="Container has visible GPU devices.",
                details=details,
                remediation=None,
                severity=0,
            )

        return CheckResult(
            id=self.id,
            title=self.title,
            category=self.category,
            status=Status.FAIL,
            message="Container detected but GPU devices are not visible.",
            details=details,
            remediation=[
                "Run container with GPU passthrough: docker run --gpus all ...",
                "Install and configure nvidia-container-toolkit on the host.",
                "Verify host can run nvidia-smi before launching containers.",
            ],
            severity=3,
        )


@register_check
class GpuPersistenceModeCheck(BaseCheck):
    id = "gpu.persistence_mode"
    title = "GPU Persistence Mode"
    category = "gpu"

    def should_run(self, context: Context) -> bool:
        nvml_result = _results(context).get("gpu.nvml_available")
        return platform.system() == "Linux" and nvml_result is not None and nvml_result.status == Status.PASS

    def run(self, context: Context) -> CheckResult:
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            count = int(pynvml.nvmlDeviceGetCount())
            modes: list[dict[str, object]] = []
            off_indices: list[int] = []

            for idx in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                mode_raw = int(pynvml.nvmlDeviceGetPersistenceMode(handle))
                mode_enabled = mode_raw == 1
                modes.append({"index": idx, "enabled": mode_enabled})
                if not mode_enabled:
                    off_indices.append(idx)

            details = {
                "persistence_modes": modes[:8],
                "off_gpu_indices": off_indices[:16],
            }

            if off_indices:
                return CheckResult(
                    id=self.id,
                    title=self.title,
                    category=self.category,
                    status=Status.WARN,
                    message="Persistence mode is disabled on one or more GPUs.",
                    details=details,
                    remediation=[
                        "Enable persistence mode: sudo nvidia-smi -pm 1",
                    ],
                    severity=1,
                )

            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.PASS,
                message="Persistence mode is enabled on detected GPUs.",
                details=details,
                remediation=None,
                severity=0,
            )
        except Exception as exc:  # noqa: BLE001
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.WARN,
                message="Could not read GPU persistence mode via NVML.",
                details={"nvml_error": f"{type(exc).__name__}: {exc}"},
                remediation=[
                    "Verify NVML access permissions and NVIDIA driver health.",
                ],
                severity=1,
            )
        finally:
            try:
                import pynvml  # type: ignore[import-not-found]

                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass


@register_check
class GpuClockThrottleReasonsCheck(BaseCheck):
    id = "gpu.clock_throttle_reasons"
    title = "GPU Clock Throttle Reasons"
    category = "gpu"

    def should_run(self, context: Context) -> bool:
        nvml_result = _results(context).get("gpu.nvml_available")
        return platform.system() == "Linux" and nvml_result is not None and nvml_result.status == Status.PASS

    def run(self, context: Context) -> CheckResult:
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            count = int(pynvml.nvmlDeviceGetCount())
            throttle_reasons: list[dict[str, object]] = []
            throttling_detected = False

            known_flags = {
                "gpu_idle": getattr(pynvml, "nvmlClocksThrottleReasonGpuIdle", 0),
                "applications_clocks_setting": getattr(pynvml, "nvmlClocksThrottleReasonApplicationsClocksSetting", 0),
                "sw_power_cap": getattr(pynvml, "nvmlClocksThrottleReasonSwPowerCap", 0),
                "hw_slowdown": getattr(pynvml, "nvmlClocksThrottleReasonHwSlowdown", 0),
                "sync_boost": getattr(pynvml, "nvmlClocksThrottleReasonSyncBoost", 0),
                "sw_thermal_slowdown": getattr(pynvml, "nvmlClocksThrottleReasonSwThermalSlowdown", 0),
                "hw_thermal_slowdown": getattr(pynvml, "nvmlClocksThrottleReasonHwThermalSlowdown", 0),
                "hw_power_brake_slowdown": getattr(pynvml, "nvmlClocksThrottleReasonHwPowerBrakeSlowdown", 0),
            }
            no_throttle_flag = getattr(pynvml, "nvmlClocksThrottleReasonNone", 0)

            for idx in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                flags = int(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle))
                reason_names = [name for name, bit in known_flags.items() if bit and (flags & bit)]
                if flags != no_throttle_flag and reason_names:
                    throttling_detected = True

                temperature_c = None
                power_watts = None
                try:
                    sensor_gpu = getattr(pynvml, "NVML_TEMPERATURE_GPU", 0)
                    temperature_c = int(pynvml.nvmlDeviceGetTemperature(handle, sensor_gpu))
                except Exception:  # noqa: BLE001
                    temperature_c = None
                try:
                    power_mw = int(pynvml.nvmlDeviceGetPowerUsage(handle))
                    power_watts = round(power_mw / 1000.0, 2)
                except Exception:  # noqa: BLE001
                    power_watts = None

                throttle_reasons.append(
                    {
                        "index": idx,
                        "flags_raw": flags,
                        "reasons": reason_names,
                        "temperature_c": temperature_c,
                        "power_watts": power_watts,
                    }
                )

            details = {"devices": throttle_reasons[:8]}

            if throttling_detected:
                return CheckResult(
                    id=self.id,
                    title=self.title,
                    category=self.category,
                    status=Status.WARN,
                    message="One or more GPUs are currently clock-throttled.",
                    details=details,
                    remediation=[
                        "Check cooling, airflow, and sustained thermal load.",
                        "Review GPU power limits and workload duty cycle.",
                    ],
                    severity=1,
                )

            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.PASS,
                message="No active GPU clock throttling reasons detected.",
                details=details,
                remediation=None,
                severity=0,
            )
        except Exception as exc:  # noqa: BLE001
            return CheckResult(
                id=self.id,
                title=self.title,
                category=self.category,
                status=Status.WARN,
                message="Could not read GPU throttle reasons via NVML.",
                details={"nvml_error": f"{type(exc).__name__}: {exc}"},
                remediation=[
                    "Verify NVML support for throttle reason queries on this platform.",
                ],
                severity=1,
            )
        finally:
            try:
                import pynvml  # type: ignore[import-not-found]

                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass


__all__ = [
    "NvidiaSmiCheck",
    "NvmlAvailableCheck",
    "NvmlDevicesCheck",
    "RuntimeGpuPassthroughCheck",
    "GpuPersistenceModeCheck",
    "GpuClockThrottleReasonsCheck",
]
