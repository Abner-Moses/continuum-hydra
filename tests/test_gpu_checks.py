from __future__ import annotations

import subprocess
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from continuum.doctor.checks.gpu import (
    GpuPersistenceModeCheck,
    NvidiaSmiCheck,
    NvmlAvailableCheck,
    NvmlDevicesCheck,
    RuntimeGpuPassthroughCheck,
)
from continuum.doctor.models import Status


class TestGpuChecks(unittest.TestCase):
    @patch("continuum.doctor.checks.gpu.platform.system", return_value="Darwin")
    def test_gpu_checks_skip_on_macos(self, _mock_system) -> None:
        self.assertFalse(NvidiaSmiCheck().should_run({}))
        self.assertFalse(NvmlAvailableCheck().should_run({}))
        self.assertFalse(NvmlDevicesCheck().should_run({}))
        self.assertFalse(RuntimeGpuPassthroughCheck().should_run({"is_container": True}))

    @patch("continuum.doctor.checks.gpu.platform.system", return_value="Linux")
    @patch("continuum.doctor.checks.gpu.shutil.which", return_value=None)
    def test_nvidia_smi_missing_is_fail(self, _mock_which, _mock_system) -> None:
        result = NvidiaSmiCheck().run({"facts": {}, "results": {}})
        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("not found", result.message.lower())

    @patch("continuum.doctor.checks.gpu.platform.system", return_value="Linux")
    @patch("continuum.doctor.checks.gpu.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("continuum.doctor.checks.gpu.subprocess.run")
    def test_nvidia_smi_success_is_pass(self, mock_run, _mock_which, _mock_system) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["nvidia-smi", "-L"],
            returncode=0,
            stdout="GPU 0: Example",
            stderr="",
        )
        ctx = {"facts": {}, "results": {}}
        result = NvidiaSmiCheck().run(ctx)
        self.assertEqual(result.status, Status.PASS)
        self.assertTrue(ctx["facts"]["nvidia_smi_ok"])

    @patch("continuum.doctor.checks.gpu.platform.system", return_value="Linux")
    def test_persistence_mode_warn_when_off(self, _mock_system) -> None:
        fake_nvml = ModuleType("pynvml")
        fake_nvml.nvmlInit = lambda: None
        fake_nvml.nvmlShutdown = lambda: None
        fake_nvml.nvmlDeviceGetCount = lambda: 1
        fake_nvml.nvmlDeviceGetHandleByIndex = lambda idx: idx
        fake_nvml.nvmlDeviceGetPersistenceMode = lambda handle: 0

        original_nvml = sys.modules.get("pynvml")
        sys.modules["pynvml"] = fake_nvml
        try:
            ctx = {
                "facts": {},
                "results": {
                    "gpu.nvml_available": type("Result", (), {"status": Status.PASS})(),
                },
            }
            result = GpuPersistenceModeCheck().run(ctx)
            self.assertEqual(result.status, Status.WARN)
        finally:
            if original_nvml is not None:
                sys.modules["pynvml"] = original_nvml
            else:
                sys.modules.pop("pynvml", None)


if __name__ == "__main__":
    unittest.main()
