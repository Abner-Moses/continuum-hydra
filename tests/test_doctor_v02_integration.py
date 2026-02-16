from __future__ import annotations

import unittest

from continuum.doctor.checks import cuda as _cuda_checks  # noqa: F401
from continuum.doctor.checks import environment as _environment_checks  # noqa: F401
from continuum.doctor.checks import gpu as _gpu_checks  # noqa: F401
from continuum.doctor.checks import pytorch as _pytorch_checks  # noqa: F401
from continuum.doctor.checks import gpu_props as _gpu_props_checks  # noqa: F401
from continuum.doctor.checks import nccl as _nccl_checks  # noqa: F401
from continuum.doctor.checks import system as _system_checks  # noqa: F401
from continuum.doctor.checks.base import BaseCheck
from continuum.doctor.models import CheckResult, Status
from continuum.doctor.runner import DoctorRunner


class CrashingCheck(BaseCheck):
    id = "test.crash"
    title = "Crash"
    category = "test"

    def run(self, context):
        raise RuntimeError("intentional")


class TestDoctorV02Integration(unittest.TestCase):
    def test_report_includes_new_registered_checks(self) -> None:
        report = DoctorRunner(hydra_version="0.2.0").run()
        payload = report.to_dict()
        ids = {item["id"] for item in payload["checks"]}

        self.assertIn("driver.nvidia_smi", ids)
        self.assertIn("gpu.nvml_available", ids)
        self.assertIn("gpu.nvml_devices", ids)
        self.assertIn("runtime.gpu_passthrough", ids)
        self.assertIn("pytorch.installed", ids)
        self.assertIn("pytorch.cuda_available", ids)
        self.assertIn("pytorch.cuda_version", ids)
        self.assertIn("system.dev_shm", ids)
        self.assertIn("cuda.driver_version", ids)
        self.assertIn("cuda.toolkit_nvcc", ids)
        self.assertIn("cuda.driver_cuda_compat", ids)
        self.assertIn("gpu.device_properties", ids)
        self.assertIn("nccl.env_config", ids)
        self.assertIn("gpu.persistence_mode", ids)

    def test_runner_converts_check_crash_to_error_result(self) -> None:
        report = DoctorRunner(hydra_version="0.2.0", checks=[CrashingCheck]).run()
        self.assertEqual(len(report.checks), 1)
        self.assertEqual(report.checks[0].status, Status.ERROR)
        self.assertEqual(report.checks[0].id, "test.crash")
        self.assertEqual(report.summary["ERROR"], 1)
        self.assertEqual(DoctorRunner.exit_code(report), 2)

    def test_context_results_map_is_populated(self) -> None:
        class MarkerCheck(BaseCheck):
            id = "test.marker"
            title = "Marker"
            category = "test"

            def run(self, context):
                assert "results" in context
                assert "facts" in context
                assert "is_container" in context
                assert "is_wsl" in context
                return CheckResult(
                    id=self.id,
                    title=self.title,
                    category=self.category,
                    status=Status.PASS,
                    message="ok",
                )

        report = DoctorRunner(hydra_version="0.2.0", checks=[MarkerCheck]).run()
        self.assertEqual(report.summary["PASS"], 1)


if __name__ == "__main__":
    unittest.main()
