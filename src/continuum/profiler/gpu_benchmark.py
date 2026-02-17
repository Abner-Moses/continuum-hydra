from __future__ import annotations

import importlib
import importlib.util
from time import perf_counter
from typing import Any


def run_gpu_benchmark(context: dict[str, Any]) -> dict[str, Any]:
    notes = context.setdefault("notes", [])

    if bool(context.get("static_only")):
        notes.append("GPU sustained benchmark skipped due to --static-only.")
        return _empty_payload()

    if bool(context.get("no_gpu")):
        notes.append("GPU sustained benchmark skipped due to --no-gpu.")
        return _empty_payload()

    if importlib.util.find_spec("torch") is None:
        notes.append("PyTorch is not installed; GPU sustained benchmark skipped.")
        return _empty_payload()

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"PyTorch import failed; GPU sustained benchmark skipped: {type(exc).__name__}: {exc}")
        return _empty_payload()

    backend = _select_backend(torch)
    if backend is None:
        notes.append("No GPU backend available (CUDA/MPS); GPU sustained benchmark skipped.")
        return _empty_payload()

    device = "cuda:0" if backend == "cuda" else "mps"
    warmup_sec = _as_positive_float(context.get("gpu_warmup"), default=2.0)
    duration_sec = _as_positive_float(context.get("gpu_duration"), default=8.0)
    requested_dtype = str(context.get("gpu_dtype", "auto")).lower().strip() or "auto"
    size_override = _as_positive_int(context.get("gpu_size"), default=None)
    size = size_override if size_override is not None else (4096 if backend == "cuda" else 2048)

    prepared = _prepare_tensors(
        torch=torch,
        backend=backend,
        device=device,
        size=size,
        requested_dtype=requested_dtype,
        notes=notes,
    )
    if prepared is None:
        return _empty_payload(backend=backend, device=device)

    dtype_name, a, b = prepared
    try:
        warmup_end = perf_counter() + warmup_sec
        while perf_counter() < warmup_end:
            _ = torch.matmul(a, b)
            _synchronize(torch, backend)

        rates: list[float] = []
        iterations = 0
        started = perf_counter()
        end_at = started + duration_sec
        while perf_counter() < end_at:
            lap_start = perf_counter()
            _ = torch.matmul(a, b)
            _synchronize(torch, backend)
            elapsed = perf_counter() - lap_start
            if elapsed > 0:
                rates.append(1.0 / elapsed)
            iterations += 1
        measured = perf_counter() - started
    except Exception as exc:  # noqa: BLE001
        notes.append(f"GPU sustained benchmark failed: {type(exc).__name__}: {exc}")
        return _empty_payload(backend=backend, device=device, dtype_name=dtype_name)

    if not rates:
        notes.append("GPU sustained benchmark collected zero valid iterations.")
        return _empty_payload(backend=backend, device=device, dtype_name=dtype_name)

    if iterations < 5:
        notes.append("GPU sustained benchmark collected fewer than 5 iterations; variance may be noisy.")

    return {
        "gpu_sustained": {
            "backend": backend,
            "device": device,
            "dtype": dtype_name,
            "mean_iter_per_sec": _round(_mean(rates)),
            "std_iter_per_sec": _round(_std(rates)),
            "p50_iter_per_sec": _round(_percentile(rates, 50.0)),
            "p95_iter_per_sec": _round(_percentile(rates, 95.0)),
            "iterations": int(iterations),
            "duration_sec": _round(measured),
        }
    }


def _select_backend(torch: Any) -> str | None:
    try:
        if bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)()):
            return "cuda"
    except Exception:
        pass
    try:
        if bool(getattr(getattr(torch, "mps", None), "is_available", lambda: False)()):
            return "mps"
    except Exception:
        pass
    return None


def _prepare_tensors(
    torch: Any,
    backend: str,
    device: str,
    size: int,
    requested_dtype: str,
    notes: list[str],
) -> tuple[str, Any, Any] | None:
    for dtype_name in _candidate_dtypes(torch, backend, requested_dtype):
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            continue
        try:
            a = torch.randn((size, size), device=device, dtype=dtype)
            b = torch.randn((size, size), device=device, dtype=dtype)
            _ = torch.matmul(a, b)
            _synchronize(torch, backend)
            return dtype_name, a, b
        except Exception:
            continue

    notes.append(
        "No compatible GPU dtype/matmul configuration found for sustained benchmark "
        f"(requested={requested_dtype}, backend={backend})."
    )
    return None


def _candidate_dtypes(torch: Any, backend: str, requested: str) -> list[str]:
    valid = {"float16", "bfloat16", "float32"}
    if requested != "auto":
        return [requested] if requested in valid else ["float32"]
    if backend == "cuda":
        return ["float16", "bfloat16", "float32"]
    return ["float16", "float32"]


def _synchronize(torch: Any, backend: str) -> None:
    if backend == "cuda":
        sync = getattr(getattr(torch, "cuda", None), "synchronize", None)
        if callable(sync):
            sync()
        return
    if backend == "mps":
        sync = getattr(getattr(torch, "mps", None), "synchronize", None)
        if callable(sync):
            sync()


def _as_positive_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if number > 0 else default


def _as_positive_int(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    try:
        number = int(value)
    except Exception:
        return default
    return number if number > 0 else default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return variance ** 0.5


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _round(value: float) -> float:
    return round(float(value), 6)


def _empty_payload(
    backend: str | None = None,
    device: str | None = None,
    dtype_name: str | None = None,
) -> dict[str, Any]:
    return {
        "gpu_sustained": {
            "backend": backend,
            "device": device,
            "dtype": dtype_name,
            "mean_iter_per_sec": None,
            "std_iter_per_sec": None,
            "p50_iter_per_sec": None,
            "p95_iter_per_sec": None,
            "iterations": None,
            "duration_sec": None,
        }
    }


__all__ = ["run_gpu_benchmark"]
