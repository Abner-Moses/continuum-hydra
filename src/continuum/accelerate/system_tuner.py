from __future__ import annotations

import os
import platform
import re
import resource
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _run_cmd(command: list[str], timeout: int = 15) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        return 1, "", f"{type(exc).__name__}: {exc}"


def detect_context() -> dict[str, Any]:
    system = platform.system().lower()
    is_linux = system == "linux"
    is_windows = system == "windows"
    is_macos = system == "darwin"
    is_root = hasattr(os, "geteuid") and os.geteuid() == 0
    nvidia_smi = shutil.which("nvidia-smi")

    return {
        "platform": system,
        "is_linux": is_linux,
        "is_windows": is_windows,
        "is_macos": is_macos,
        "is_root": bool(is_root),
        "nvidia_smi": nvidia_smi,
        "nvidia_present": nvidia_smi is not None,
    }


def _read_cpu_governor() -> str | None:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:  # noqa: BLE001
        return None


def _read_swappiness() -> int | None:
    path = Path("/proc/sys/vm/swappiness")
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8", errors="ignore").strip())
    except Exception:  # noqa: BLE001
        return None


def _read_nvidia_persistence() -> str | None:
    code, out, _err = _run_cmd(["nvidia-smi", "-q", "-d", "PERFORMANCE"])
    if code != 0:
        return None
    match = re.search(r"Persistence Mode\s*:\s*(Enabled|Disabled)", out, re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).lower()


def _read_windows_power_plan() -> str | None:
    code, out, _err = _run_cmd(["powercfg", "/getactivescheme"])
    if code != 0:
        return None
    match = re.search(r"([0-9a-fA-F\-]{36})", out)
    return match.group(1) if match else None


def capture_previous_state(ctx: dict[str, Any], cpu_only: bool, gpu_only: bool) -> dict[str, Any]:
    state: dict[str, Any] = {
        "nice": os.nice(0) if hasattr(os, "nice") else None,
        "rlimit_nofile": None,
    }

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        state["rlimit_nofile"] = {"soft": int(soft), "hard": int(hard)}
    except Exception:  # noqa: BLE001
        state["rlimit_nofile"] = None

    if ctx["is_linux"] and not gpu_only:
        state["cpu_governor"] = _read_cpu_governor()
        state["swappiness"] = _read_swappiness()

    if ctx["nvidia_present"] and not cpu_only:
        state["nvidia_persistence_mode"] = _read_nvidia_persistence()

    if ctx["is_windows"] and not gpu_only:
        try:
            import psutil  # type: ignore

            state["process_priority"] = int(psutil.Process().nice())
        except Exception:  # noqa: BLE001
            state["process_priority"] = None
        state["power_plan_guid"] = _read_windows_power_plan()

    return state


def _change(
    *,
    name: str,
    result: str,
    message: str,
    requires_root: bool,
    category: str,
    command: str | None = None,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "result": result,
        "message": message,
        "requires_root": requires_root,
        "category": category,
    }
    if command is not None:
        payload["command"] = command
    return payload


def _sorted_changes(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(changes, key=lambda item: item.get("name", ""))


def apply_acceleration(
    ctx: dict[str, Any],
    previous_state: dict[str, Any],
    dry_run: bool,
    cpu_only: bool,
    gpu_only: bool,
    allow_risky: bool,
) -> tuple[list[dict[str, Any]], list[str], list[str], dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    failures: list[str] = []
    applied_actions: list[str] = []
    applied_previous_state: dict[str, Any] = {}

    def record_applied(action_name: str, previous_key: str) -> None:
        if action_name not in applied_actions:
            applied_actions.append(action_name)
        if previous_key in previous_state:
            applied_previous_state[previous_key] = previous_state.get(previous_key)

    if ctx["is_linux"] and not gpu_only:
        governor = previous_state.get("cpu_governor")
        if governor is None:
            changes.append(_change(name="cpu_governor", result="skipped", message="governor path unavailable", requires_root=True, category="cpu"))
        elif dry_run:
            changes.append(_change(name="cpu_governor", result="planned", message="would set governor to performance", requires_root=True, category="cpu", command="cpupower frequency-set -g performance"))
        elif not ctx["is_root"]:
            changes.append(_change(name="cpu_governor", result="skipped", message="root required", requires_root=True, category="cpu"))
        elif shutil.which("cpupower") is None:
            changes.append(_change(name="cpu_governor", result="skipped", message="cpupower not installed", requires_root=True, category="cpu"))
        else:
            code, _out, err = _run_cmd(["cpupower", "frequency-set", "-g", "performance"])
            if code == 0:
                changes.append(_change(name="cpu_governor", result="applied", message="set to performance", requires_root=True, category="cpu", command="cpupower frequency-set -g performance"))
                record_applied("cpu_governor", "cpu_governor")
            else:
                failures.append(f"cpu_governor: {err or 'unknown error'}")
                changes.append(_change(name="cpu_governor", result="skipped", message=f"failed: {err or 'unknown error'}", requires_root=True, category="cpu", command="cpupower frequency-set -g performance"))

        if hasattr(os, "nice"):
            if dry_run:
                changes.append(_change(name="process_nice", result="planned", message="would raise process priority (nice -5)", requires_root=False, category="cpu"))
            else:
                try:
                    os.nice(-5)
                    changes.append(_change(name="process_nice", result="applied", message="raised process priority", requires_root=False, category="cpu"))
                    record_applied("process_nice", "nice")
                except Exception as exc:  # noqa: BLE001
                    changes.append(_change(name="process_nice", result="skipped", message=f"insufficient permission: {exc}", requires_root=False, category="cpu"))

        swappiness = previous_state.get("swappiness")
        if swappiness is None:
            changes.append(_change(name="swappiness", result="skipped", message="swappiness not available", requires_root=True, category="cpu", command="sysctl -w vm.swappiness=10"))
        elif not allow_risky:
            planned_or_not = "planned" if dry_run else "not-applied"
            changes.append(_change(name="swappiness", result=planned_or_not, message="opt-in required for risky tunable", requires_root=True, category="cpu", command="sysctl -w vm.swappiness=10"))
        elif dry_run:
            changes.append(_change(name="swappiness", result="planned", message="would set vm.swappiness=10", requires_root=True, category="cpu", command="sysctl -w vm.swappiness=10"))
        elif not ctx["is_root"]:
            changes.append(_change(name="swappiness", result="skipped", message="root required", requires_root=True, category="cpu", command="sysctl -w vm.swappiness=10"))
        else:
            code, _out, err = _run_cmd(["sysctl", "-w", "vm.swappiness=10"])
            if code == 0:
                changes.append(_change(name="swappiness", result="applied", message="vm.swappiness set to 10", requires_root=True, category="cpu", command="sysctl -w vm.swappiness=10"))
                record_applied("swappiness", "swappiness")
            else:
                failures.append(f"swappiness: {err or 'unknown error'}")
                changes.append(_change(name="swappiness", result="skipped", message=f"failed: {err or 'unknown error'}", requires_root=True, category="cpu", command="sysctl -w vm.swappiness=10"))

        limits = previous_state.get("rlimit_nofile") or {}
        soft = limits.get("soft")
        hard = limits.get("hard")
        if soft is None or hard is None:
            changes.append(_change(name="ulimit_nofile", result="skipped", message="rlimit unavailable", requires_root=False, category="cpu"))
        elif dry_run:
            changes.append(_change(name="ulimit_nofile", result="planned", message="would raise soft open-file limit", requires_root=False, category="cpu"))
        else:
            target = min(int(hard), max(int(soft), 65535))
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (target, int(hard)))
                changes.append(_change(name="ulimit_nofile", result="applied", message=f"soft limit set to {target}", requires_root=False, category="cpu"))
                record_applied("ulimit_nofile", "rlimit_nofile")
            except Exception as exc:  # noqa: BLE001
                changes.append(_change(name="ulimit_nofile", result="skipped", message=f"unable to set rlimit: {exc}", requires_root=False, category="cpu"))

    if ctx["is_windows"] and not gpu_only:
        if dry_run:
            changes.append(_change(name="windows_process_priority", result="planned", message="would set HIGH priority", requires_root=False, category="cpu"))
        else:
            try:
                import psutil  # type: ignore

                process = psutil.Process()
                process.nice(psutil.HIGH_PRIORITY_CLASS)
                changes.append(_change(name="windows_process_priority", result="applied", message="set HIGH priority", requires_root=False, category="cpu"))
                record_applied("windows_process_priority", "process_priority")
            except Exception as exc:  # noqa: BLE001
                changes.append(_change(name="windows_process_priority", result="skipped", message=f"{exc}", requires_root=False, category="cpu"))

        if not allow_risky:
            changes.append(_change(name="windows_power_plan", result="not-applied", message="opt-in required for risky tunable", requires_root=False, category="cpu"))
        elif dry_run:
            changes.append(_change(name="windows_power_plan", result="planned", message="would set high performance power plan", requires_root=False, category="cpu", command="powercfg /setactive SCHEME_MIN"))
        else:
            code, _out, err = _run_cmd(["powercfg", "/setactive", "SCHEME_MIN"])
            if code == 0:
                changes.append(_change(name="windows_power_plan", result="applied", message="set high performance power plan", requires_root=False, category="cpu", command="powercfg /setactive SCHEME_MIN"))
                record_applied("windows_power_plan", "power_plan_guid")
            else:
                changes.append(_change(name="windows_power_plan", result="skipped", message=err or "unable to change power plan", requires_root=False, category="cpu", command="powercfg /setactive SCHEME_MIN"))

    if ctx["nvidia_present"] and not cpu_only:
        if dry_run:
            changes.append(_change(name="nvidia_persistence", result="planned", message="would enable persistence mode", requires_root=True, category="gpu", command="nvidia-smi -pm 1"))
        elif not ctx["is_root"]:
            changes.append(_change(name="nvidia_persistence", result="skipped", message="root/admin may be required", requires_root=True, category="gpu", command="nvidia-smi -pm 1"))
        else:
            code, _out, err = _run_cmd(["nvidia-smi", "-pm", "1"])
            if code == 0:
                changes.append(_change(name="nvidia_persistence", result="applied", message="enabled persistence mode", requires_root=True, category="gpu", command="nvidia-smi -pm 1"))
                record_applied("nvidia_persistence", "nvidia_persistence_mode")
            else:
                changes.append(_change(name="nvidia_persistence", result="skipped", message=err or "unable to enable persistence mode", requires_root=True, category="gpu", command="nvidia-smi -pm 1"))
    elif not gpu_only:
        changes.append(_change(name="nvidia_persistence", result="skipped", message="nvidia-smi not found", requires_root=True, category="gpu", command="nvidia-smi -pm 1"))

    return _sorted_changes(changes), failures, sorted(applied_actions), applied_previous_state


def restore_acceleration(
    ctx: dict[str, Any],
    previous_state: dict[str, Any],
    applied_actions: list[str],
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    changes: list[dict[str, Any]] = []
    failures: list[str] = []
    applied_set = set(applied_actions)

    def mark_not_applied(name: str, category: str) -> None:
        changes.append(_change(name=name, result="not-applied", message="was not applied during --on", requires_root=False, category=category))

    if "cpu_governor" in applied_set:
        governor = previous_state.get("cpu_governor")
        if governor:
            if dry_run:
                changes.append(_change(name="cpu_governor", result="planned", message=f"would restore governor={governor}", requires_root=True, category="cpu"))
            elif ctx["is_root"] and shutil.which("cpupower"):
                code, _out, err = _run_cmd(["cpupower", "frequency-set", "-g", str(governor)])
                if code == 0:
                    changes.append(_change(name="cpu_governor", result="restored", message=f"restored governor={governor}", requires_root=True, category="cpu"))
                else:
                    failures.append(f"cpu_governor restore: {err or 'unknown error'}")
                    changes.append(_change(name="cpu_governor", result="skipped", message=f"failed: {err or 'unknown error'}", requires_root=True, category="cpu"))
            else:
                changes.append(_change(name="cpu_governor", result="skipped", message="root/cpupower unavailable for restore", requires_root=True, category="cpu"))
    else:
        mark_not_applied("cpu_governor", "cpu")

    if "swappiness" in applied_set:
        swappiness = previous_state.get("swappiness")
        if swappiness is not None:
            if dry_run:
                changes.append(_change(name="swappiness", result="planned", message=f"would restore vm.swappiness={swappiness}", requires_root=True, category="cpu", command=f"sysctl -w vm.swappiness={int(swappiness)}"))
            elif ctx["is_root"]:
                code, _out, err = _run_cmd(["sysctl", "-w", f"vm.swappiness={int(swappiness)}"])
                if code == 0:
                    changes.append(_change(name="swappiness", result="restored", message=f"restored vm.swappiness={swappiness}", requires_root=True, category="cpu", command=f"sysctl -w vm.swappiness={int(swappiness)}"))
                else:
                    failures.append(f"swappiness restore: {err or 'unknown error'}")
                    changes.append(_change(name="swappiness", result="skipped", message=f"failed: {err or 'unknown error'}", requires_root=True, category="cpu"))
            else:
                changes.append(_change(name="swappiness", result="skipped", message="root required for restore", requires_root=True, category="cpu"))
    else:
        mark_not_applied("swappiness", "cpu")

    if "ulimit_nofile" in applied_set:
        limits = previous_state.get("rlimit_nofile") or {}
        soft = limits.get("soft")
        hard = limits.get("hard")
        if soft is not None and hard is not None:
            if dry_run:
                changes.append(_change(name="ulimit_nofile", result="planned", message=f"would restore soft={soft}, hard={hard}", requires_root=False, category="cpu"))
            else:
                try:
                    resource.setrlimit(resource.RLIMIT_NOFILE, (int(soft), int(hard)))
                    changes.append(_change(name="ulimit_nofile", result="restored", message=f"restored soft={soft}, hard={hard}", requires_root=False, category="cpu"))
                except Exception as exc:  # noqa: BLE001
                    changes.append(_change(name="ulimit_nofile", result="skipped", message=f"unable to restore rlimit: {exc}", requires_root=False, category="cpu"))
    else:
        mark_not_applied("ulimit_nofile", "cpu")

    if "nvidia_persistence" in applied_set:
        previous = previous_state.get("nvidia_persistence_mode")
        if previous in {"enabled", "disabled"}:
            target = "1" if previous == "enabled" else "0"
            if dry_run:
                changes.append(_change(name="nvidia_persistence", result="planned", message=f"would restore persistence={previous}", requires_root=True, category="gpu", command=f"nvidia-smi -pm {target}"))
            elif ctx["is_root"]:
                code, _out, err = _run_cmd(["nvidia-smi", "-pm", target])
                if code == 0:
                    changes.append(_change(name="nvidia_persistence", result="restored", message=f"restored persistence={previous}", requires_root=True, category="gpu", command=f"nvidia-smi -pm {target}"))
                else:
                    failures.append(f"nvidia_persistence restore: {err or 'unknown error'}")
                    changes.append(_change(name="nvidia_persistence", result="skipped", message=f"failed: {err or 'unknown error'}", requires_root=True, category="gpu"))
            else:
                changes.append(_change(name="nvidia_persistence", result="skipped", message="root/admin may be required for restore", requires_root=True, category="gpu"))
    else:
        mark_not_applied("nvidia_persistence", "gpu")

    if "windows_process_priority" in applied_set:
        priority = previous_state.get("process_priority")
        if priority is not None:
            if dry_run:
                changes.append(_change(name="windows_process_priority", result="planned", message=f"would restore priority={priority}", requires_root=False, category="cpu"))
            else:
                try:
                    import psutil  # type: ignore

                    psutil.Process().nice(int(priority))
                    changes.append(_change(name="windows_process_priority", result="restored", message=f"restored priority={priority}", requires_root=False, category="cpu"))
                except Exception as exc:  # noqa: BLE001
                    changes.append(_change(name="windows_process_priority", result="skipped", message=f"unable to restore priority: {exc}", requires_root=False, category="cpu"))

    if "windows_power_plan" in applied_set:
        power_plan = previous_state.get("power_plan_guid")
        if power_plan:
            if dry_run:
                changes.append(_change(name="windows_power_plan", result="planned", message=f"would restore power plan={power_plan}", requires_root=False, category="cpu", command=f"powercfg /setactive {power_plan}"))
            else:
                code, _out, err = _run_cmd(["powercfg", "/setactive", str(power_plan)])
                if code == 0:
                    changes.append(_change(name="windows_power_plan", result="restored", message=f"restored power plan={power_plan}", requires_root=False, category="cpu", command=f"powercfg /setactive {power_plan}"))
                else:
                    changes.append(_change(name="windows_power_plan", result="skipped", message=err or "unable to restore power plan", requires_root=False, category="cpu"))

    if "process_nice" in applied_set:
        previous_nice = previous_state.get("nice")
        if previous_nice is not None and hasattr(os, "nice"):
            if dry_run:
                changes.append(_change(name="process_nice", result="planned", message=f"would restore nice={previous_nice}", requires_root=False, category="cpu"))
            else:
                try:
                    delta = int(previous_nice) - os.nice(0)
                    if delta != 0:
                        os.nice(delta)
                    changes.append(_change(name="process_nice", result="restored", message=f"restored nice={previous_nice}", requires_root=False, category="cpu"))
                except Exception as exc:  # noqa: BLE001
                    changes.append(_change(name="process_nice", result="skipped", message=f"unable to restore nice: {exc}", requires_root=False, category="cpu"))

    return _sorted_changes(changes), failures


__all__ = [
    "detect_context",
    "capture_previous_state",
    "apply_acceleration",
    "restore_acceleration",
]
