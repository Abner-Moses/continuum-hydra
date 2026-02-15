from __future__ import annotations

import platform
import sys
from pathlib import Path


_PROC_VERSION = Path('/proc/version')
_PROC_1_CGROUP = Path('/proc/1/cgroup')
_DOCKER_ENV = Path('/.dockerenv')
_PODMAN_ENV = Path('/run/.containerenv')


def get_hostname() -> str:
    return platform.node()


def is_wsl() -> bool:
    if platform.system() != 'Linux':
        return False

    try:
        content = _PROC_VERSION.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return False

    return 'microsoft' in content.lower()


def is_container() -> bool:
    if platform.system() != 'Linux':
        return False

    if _DOCKER_ENV.exists() or _PODMAN_ENV.exists():
        return True

    try:
        cgroup = _PROC_1_CGROUP.read_text(encoding='utf-8', errors='ignore').lower()
    except OSError:
        return False

    return any(token in cgroup for token in ('docker', 'containerd'))


def get_os_string() -> str:
    return f"{platform.system()} {platform.release()}"


def get_python_version_string() -> str:
    return platform.python_version()


def get_python_executable() -> str:
    return sys.executable


__all__ = [
    'get_hostname',
    'is_wsl',
    'is_container',
    'get_os_string',
    'get_python_version_string',
    'get_python_executable',
]
