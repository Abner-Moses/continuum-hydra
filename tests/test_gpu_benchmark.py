from __future__ import annotations

import importlib
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from continuum.profiler.gpu_benchmark import run_gpu_benchmark


class _FakeTensor:
    pass


class _FakeTorch:
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"

    def __init__(self, *, cuda_available: bool, mps_available: bool) -> None:
        self.cuda = SimpleNamespace(
            is_available=lambda: cuda_available,
            synchronize=lambda: None,
        )
        self.mps = SimpleNamespace(
            is_available=lambda: mps_available,
            synchronize=lambda: None,
        )

    def randn(self, shape, device=None, dtype=None):  # noqa: ANN001, ANN201
        return _FakeTensor()

    def matmul(self, a, b):  # noqa: ANN001, ANN201
        return _FakeTensor()


class TestGpuBenchmark(unittest.TestCase):
    def test_torch_missing_returns_null_payload(self) -> None:
        ctx = {"notes": []}
        with patch("continuum.profiler.gpu_benchmark.importlib.util.find_spec", return_value=None):
            result = run_gpu_benchmark(ctx)

        payload = result["gpu_sustained"]
        self.assertIsNone(payload["mean_iter_per_sec"])
        self.assertIsNone(payload["backend"])
        self.assertTrue(any("pytorch" in note.lower() for note in ctx["notes"]))

    def test_torch_present_but_no_backend(self) -> None:
        fake_torch = _FakeTorch(cuda_available=False, mps_available=False)
        real_import_module = importlib.import_module

        def _import_module(name: str):  # noqa: ANN202
            if name == "torch":
                return fake_torch
            return real_import_module(name)

        with patch("continuum.profiler.gpu_benchmark.importlib.util.find_spec", return_value=object()):
            with patch("continuum.profiler.gpu_benchmark.importlib.import_module", side_effect=_import_module):
                result = run_gpu_benchmark({"notes": []})

        payload = result["gpu_sustained"]
        self.assertIsNone(payload["mean_iter_per_sec"])
        self.assertIsNone(payload["backend"])

    def test_static_only_skips_benchmark(self) -> None:
        ctx = {"static_only": True, "notes": []}
        result = run_gpu_benchmark(ctx)
        payload = result["gpu_sustained"]

        self.assertIsNone(payload["mean_iter_per_sec"])
        self.assertIsNone(payload["iterations"])
        self.assertTrue(any("static-only" in note.lower() for note in ctx["notes"]))

    def test_deterministic_structure_validation(self) -> None:
        fake_torch = _FakeTorch(cuda_available=True, mps_available=False)
        ticks = itertools.count(step=0.01)
        real_import_module = importlib.import_module

        def _import_module(name: str):  # noqa: ANN202
            if name == "torch":
                return fake_torch
            return real_import_module(name)

        with patch("continuum.profiler.gpu_benchmark.importlib.util.find_spec", return_value=object()):
            with patch("continuum.profiler.gpu_benchmark.importlib.import_module", side_effect=_import_module):
                with patch("continuum.profiler.gpu_benchmark.perf_counter", side_effect=lambda: next(ticks)):
                    result = run_gpu_benchmark(
                        {
                            "notes": [],
                            "gpu_warmup": 0.0,
                            "gpu_duration": 0.1,
                            "gpu_size": 128,
                            "gpu_dtype": "auto",
                        }
                    )

        payload = result["gpu_sustained"]
        self.assertIn(type(payload["backend"]), {str, type(None)})
        self.assertIn(type(payload["device"]), {str, type(None)})
        self.assertIn(type(payload["dtype"]), {str, type(None)})
        self.assertIn(type(payload["mean_iter_per_sec"]), {float, type(None)})
        self.assertIn(type(payload["std_iter_per_sec"]), {float, type(None)})
        self.assertIn(type(payload["p50_iter_per_sec"]), {float, type(None)})
        self.assertIn(type(payload["p95_iter_per_sec"]), {float, type(None)})
        self.assertIn(type(payload["iterations"]), {int, type(None)})
        self.assertIn(type(payload["duration_sec"]), {float, type(None)})


if __name__ == "__main__":
    unittest.main()
