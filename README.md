# continuum-hydra

Hydra is a performance-first ML systems toolkit built under the Continuum infrastructure initiative.

Hydra Doctor provides preflight diagnostics for Python/runtime, NVIDIA driver/NVML/GPU visibility, CUDA compatibility/toolkit hints, PyTorch readiness, and soft NCCL checks before training begins.

Hydra is designed to reduce silent training failures, version mismatches, and hidden performance regressions in GPU-based workloads.

## Design Principles

- Structured diagnostics with explicit pass/warn/fail status
- No silent system modifications
- Reproducible structured reports
- Infrastructure-grade transparency

Hydra Doctor is **safe and side-effect free**. It reads environment/system info, prints a report, and optionally writes a JSON file.

### Prerequisites

- Python **3.10+**
- `pip` (recommended: latest)
- For full GPU diagnostics on Linux/Windows: NVIDIA drivers (`nvidia-smi`), NVML (`pynvml`), PyTorch (`torch`)

---

## Run

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate            # macOS/Linux
# .\.venv\Scripts\activate           # Windows PowerShell

python -m pip install -U pip
python -m pip install -e .

continuum doctor
continuum doctor --json
continuum doctor --export /tmp/reports
continuum doctor --no-write
continuum doctor --only gpu,cuda
continuum doctor --exclude nccl.env_config
continuum doctor --list-checks
continuum doctor --deterministic
```

## What Doctor Checks

- Environment: Python version, virtualenv/conda isolation, runtime container/WSL detection
- Driver/GPU: `nvidia-smi`, NVML init, NVML device enumeration, container GPU passthrough visibility
- CUDA: driver version detection, `nvcc` toolkit detection, torch CUDA version, driver/CUDA compatibility matrix, runtime path hints
- PyTorch: installation detection, `torch.cuda.is_available()`, torch CUDA/cuDNN metadata
- GPU properties/health: compute capability and tensor-core readiness hint, persistence mode, active throttle reasons
- NCCL (soft checks): environment variable sanity and torch NCCL backend availability (no distributed collectives executed)
- System: Linux `/dev/shm` capacity checks

Linux-first behavior:
- Linux: full check set runs (subject to environment and dependencies)
- Windows: environment + driver/GPU/CUDA/PyTorch checks run; Linux-only checks are skipped
- macOS: NVIDIA/CUDA/NCCL Linux/Windows checks are skipped safely; generic environment checks still run

## Outputs

- Human-readable Rich table is printed to console.
- JSON report is written by default to:
  - `.hydra/reports/doctor_YYYYMMDD_HHMMSS.json`
- Use `--no-write` to disable file writing.
- Use `--json` to also print JSON to stdout.
- Use `--export <dir>` to change the output directory.
- Use `--only <ids_or_categories>` to include a subset (comma-separated).
- Use `--exclude <check_ids>` to skip specific checks (comma-separated).
- Use `--list-checks` to print `check_id`, `category`, and `title` without executing checks.
- Use `--deterministic` for stable CI snapshots (`timestamp_utc=1970-01-01T00:00:00Z`, all durations `0.0`).

Top-level JSON shape:

```json
{
  "schema_version": "1.0.0",
  "environment": { "...": "..." },
  "checks": [],
  "summary": { "PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0 },
  "overall_status": "healthy",
  "total_duration_ms": 0.0
}
```

## Exit Codes

| Exit code | Meaning |
| --- | --- |
| `0` | Healthy (no warnings/failures/errors) |
| `1` | Warnings present |
| `2` | Failed checks or check errors |
| `4` | Tool-level crash/unhandled failure |

## Testing

Offline-friendly unittest command:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v
```

Fresh-user smoke install/run:

```bash
./scripts/smoke_install_and_run.sh
```

Manual copy/paste smoke flow:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
continuum doctor --list-checks
continuum doctor --deterministic --json --no-write
continuum doctor
ls -1 .hydra/reports/doctor_*.json | tail -n 1
```
![Status](https://img.shields.io/badge/status-hydra--doctor--development-orange)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
