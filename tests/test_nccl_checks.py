from __future__ import annotations

import os
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from continuum.doctor.checks.nccl import NcclEnvConfigCheck, NcclTorchBackendCheck
from continuum.doctor.models import CheckResult, Status


class TestNcclChecks(unittest.TestCase):
    @patch("continuum.doctor.checks.nccl.platform.system", return_value="Linux")
    def test_env_config_warns_on_suspicious_vars(self, _mock_system) -> None:
        check = NcclEnvConfigCheck()
        with patch.dict(
            os.environ,
            {"NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "lo"},
            clear=False,
        ):
            result = check.run({"facts": {"gpu_count": 2}, "is_container": False})
        self.assertEqual(result.status, Status.WARN)

    @patch("continuum.doctor.checks.nccl.platform.system", return_value="Linux")
    def test_torch_backend_warns_when_multi_gpu_and_nccl_missing(self, _mock_system) -> None:
        fake_dist = ModuleType("torch.distributed")
        fake_dist.is_available = lambda: True
        fake_dist.is_nccl_available = lambda: False

        fake_torch = ModuleType("torch")
        fake_torch.distributed = fake_dist

        original_torch = sys.modules.get("torch")
        original_dist = sys.modules.get("torch.distributed")
        sys.modules["torch"] = fake_torch
        sys.modules["torch.distributed"] = fake_dist
        try:
            ctx = {
                "facts": {"gpu_count": 2},
                "results": {
                    "pytorch.installed": CheckResult(
                        id="pytorch.installed",
                        title="PyTorch Installation",
                        category="pytorch",
                        status=Status.PASS,
                        message="ok",
                    )
                },
            }
            result = NcclTorchBackendCheck().run(ctx)
            self.assertEqual(result.status, Status.WARN)
        finally:
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            else:
                sys.modules.pop("torch", None)
            if original_dist is not None:
                sys.modules["torch.distributed"] = original_dist
            else:
                sys.modules.pop("torch.distributed", None)


if __name__ == "__main__":
    unittest.main()
