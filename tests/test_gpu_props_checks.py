from __future__ import annotations

import sys
import unittest
from types import ModuleType, SimpleNamespace

from continuum.doctor.checks.gpu_props import GpuDevicePropertiesCheck
from continuum.doctor.models import CheckResult, Status


class _FakeCuda:
    def __init__(self, props):
        self._props = props

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return len(self._props)

    def get_device_properties(self, index: int):
        return self._props[index]


class TestGpuPropsChecks(unittest.TestCase):
    def test_warn_on_low_compute_capability(self) -> None:
        fake_torch = ModuleType("torch")
        fake_torch.cuda = _FakeCuda(
            [
                SimpleNamespace(name="GPU-A", major=6, minor=1, total_memory=8 * 1024**3, multi_processor_count=20),
                SimpleNamespace(name="GPU-B", major=8, minor=0, total_memory=24 * 1024**3, multi_processor_count=80),
            ]
        )

        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            ctx = {
                "results": {
                    "pytorch.installed": CheckResult(
                        id="pytorch.installed",
                        title="PyTorch Installation",
                        category="pytorch",
                        status=Status.PASS,
                        message="ok",
                    )
                },
                "facts": {},
            }
            result = GpuDevicePropertiesCheck().run(ctx)
            self.assertEqual(result.status, Status.WARN)
            self.assertEqual(result.details["device_count"], 2)
        finally:
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            else:
                sys.modules.pop("torch", None)


if __name__ == "__main__":
    unittest.main()
