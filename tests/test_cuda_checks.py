from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from continuum.doctor.checks.cuda import (
    CudaDriverCompatCheck,
    CudaDriverVersionCheck,
    CudaToolkitNvccCheck,
)
from continuum.doctor.models import Status


class TestCudaChecks(unittest.TestCase):
    @patch("continuum.doctor.checks.cuda._driver_version_from_nvml", return_value=None)
    @patch("continuum.doctor.checks.cuda.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("continuum.doctor.checks.cuda.subprocess.run")
    def test_driver_version_parsing_from_nvidia_smi(self, mock_run, _mock_which, _mock_nvml) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            returncode=0,
            stdout="550.54.14\n",
            stderr="",
        )
        ctx = {"facts": {}, "results": {}}
        result = CudaDriverVersionCheck().run(ctx)
        self.assertEqual(result.status, Status.PASS)
        self.assertEqual(result.details["driver_version"], "550.54.14")

    def test_driver_cuda_compat_matrix_fail_warn_pass(self) -> None:
        check = CudaDriverCompatCheck()

        fail_result = check.run({"facts": {"driver_version": "520.00.00", "torch_cuda_version": "12.4"}})
        self.assertEqual(fail_result.status, Status.FAIL)

        warn_result = check.run({"facts": {"driver_version": "550.54.14", "torch_cuda_version": "13.0"}})
        self.assertEqual(warn_result.status, Status.WARN)

        pass_result = check.run({"facts": {"driver_version": "550.80.00", "torch_cuda_version": "12.4"}})
        self.assertEqual(pass_result.status, Status.PASS)

    @patch("continuum.doctor.checks.cuda.shutil.which", return_value="/usr/local/cuda/bin/nvcc")
    @patch("continuum.doctor.checks.cuda.subprocess.run")
    def test_nvcc_detection_parses_version(self, mock_run, _mock_which) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["nvcc", "--version"],
            returncode=0,
            stdout="Cuda compilation tools, release 12.4, V12.4.99",
            stderr="",
        )
        ctx = {"facts": {}, "results": {}}
        result = CudaToolkitNvccCheck().run(ctx)
        self.assertEqual(result.status, Status.PASS)
        self.assertEqual(result.details["nvcc_version"], "12.4")


if __name__ == "__main__":
    unittest.main()
