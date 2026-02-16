from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from continuum.doctor.checks.system import DevShmCheck
from continuum.doctor.models import Status

_GIB = 1024**3


class TestSystemChecks(unittest.TestCase):
    @patch("continuum.doctor.checks.system.platform.system", return_value="Darwin")
    def test_dev_shm_should_run_linux_only(self, _mock_system) -> None:
        self.assertFalse(DevShmCheck().should_run({}))

    @patch("continuum.doctor.checks.system.platform.system", return_value="Linux")
    @patch("continuum.doctor.checks.system.os.statvfs")
    def test_dev_shm_warn_and_fail_thresholds(self, mock_statvfs, _mock_system) -> None:
        mock_statvfs.return_value = SimpleNamespace(f_frsize=1, f_blocks=_GIB // 2)
        fail_result = DevShmCheck().run({})
        self.assertEqual(fail_result.status, Status.FAIL)

        mock_statvfs.return_value = SimpleNamespace(f_frsize=1, f_blocks=2 * _GIB)
        warn_result = DevShmCheck().run({})
        self.assertEqual(warn_result.status, Status.WARN)

        mock_statvfs.return_value = SimpleNamespace(f_frsize=1, f_blocks=10 * _GIB)
        pass_result = DevShmCheck().run({})
        self.assertEqual(pass_result.status, Status.PASS)


if __name__ == "__main__":
    unittest.main()
