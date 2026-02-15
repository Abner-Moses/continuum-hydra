from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar, runtime_checkable

from continuum.doctor.models import CheckResult

Context = dict[str, Any]


@runtime_checkable
class Check(Protocol):
    id: str
    title: str
    category: str

    def run(self, context: Context) -> CheckResult:
        ...


class BaseCheck(ABC):
    id: str
    title: str
    category: str

    def should_run(self, context: Context) -> bool:
        return True

    @abstractmethod
    def run(self, context: Context) -> CheckResult:
        raise NotImplementedError


CheckClass = type[BaseCheck]
_CheckT = TypeVar("_CheckT", bound=CheckClass)
_CHECK_REGISTRY: list[CheckClass] = []


def register_check(check_cls: _CheckT) -> _CheckT:
    if check_cls not in _CHECK_REGISTRY:
        _CHECK_REGISTRY.append(check_cls)
    return check_cls


def list_checks() -> list[CheckClass]:
    return list(_CHECK_REGISTRY)


__all__ = [
    "Context",
    "Check",
    "BaseCheck",
    "register_check",
    "list_checks",
]
