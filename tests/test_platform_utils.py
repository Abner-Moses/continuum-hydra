from __future__ import annotations

import unittest
from unittest.mock import patch

from continuum.doctor.utils import platform as platform_utils


class _FakePath:
    def __init__(self, *, exists_value: bool = False, read_text_value: str = "") -> None:
        self._exists_value = exists_value
        self._read_text_value = read_text_value

    def exists(self) -> bool:
        return self._exists_value

    def read_text(self, encoding: str = "utf-8", errors: str = "ignore") -> str:
        return self._read_text_value


class TestPlatformUtils(unittest.TestCase):
    def test_is_wsl_false_on_non_linux(self) -> None:
        with patch.object(platform_utils.platform, "system", return_value="Darwin"):
            self.assertFalse(platform_utils.is_wsl())

    def test_is_container_true_when_marker_file_exists(self) -> None:
        with patch.object(platform_utils.platform, "system", return_value="Linux"):
            with patch.object(platform_utils, "_DOCKER_ENV", _FakePath(exists_value=True)):
                self.assertTrue(platform_utils.is_container())

    def test_is_container_true_for_kubernetes_cgroup(self) -> None:
        with patch.object(platform_utils.platform, "system", return_value="Linux"):
            with patch.object(platform_utils, "_DOCKER_ENV", _FakePath(exists_value=False)):
                with patch.object(platform_utils, "_PODMAN_ENV", _FakePath(exists_value=False)):
                    with patch.object(
                        platform_utils,
                        "_PROC_1_CGROUP",
                        _FakePath(read_text_value="1:name=systemd:/kubepods/besteffort/podabc123"),
                    ):
                        self.assertTrue(platform_utils.is_container())


if __name__ == "__main__":
    unittest.main()
