from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from continuum.doctor.checks.pytorch import PytorchInstalledCheck
from continuum.doctor.models import Status


class TestPytorchChecks(unittest.TestCase):
    @patch("continuum.doctor.checks.pytorch.importlib.util.find_spec", return_value=None)
    def test_pytorch_installed_fail_when_missing(self, _mock_find_spec) -> None:
        ctx = {"facts": {}, "results": {}}
        result = PytorchInstalledCheck().run(ctx)
        self.assertEqual(result.status, Status.FAIL)
        self.assertFalse(ctx["facts"]["torch_installed"])

    @patch(
        "continuum.doctor.checks.pytorch.importlib.util.find_spec",
        return_value=SimpleNamespace(origin="/venv/lib/python/site-packages/torch/__init__.py"),
    )
    def test_pytorch_installed_pass_when_present(self, _mock_find_spec) -> None:
        ctx = {"facts": {}, "results": {}}
        result = PytorchInstalledCheck().run(ctx)
        self.assertEqual(result.status, Status.PASS)
        self.assertTrue(ctx["facts"]["torch_installed"])


if __name__ == "__main__":
    unittest.main()
