# continuum-hydra

Hydra is a performance-first ML systems toolkit built under the Continuum infrastructure initiative.

The first module, Hydra Doctor, provides deterministic diagnostics for GPU, CUDA, and PyTorch environments before training begins.

Hydra is designed to reduce silent training failures, version mismatches, and hidden performance regressions in GPU-based workloads.

## Design Principles

- Deterministic diagnostics
- No silent system modifications
- Reproducible structured reports
- Infrastructure-grade transparency

## Testing

Hydra Doctor is designed to be **safe and side-effect free**. It reads environment/system info, prints a report, and optionally writes a JSON file.

### Prerequisites

- Python **3.10+**
- `pip` (recommended: latest)
- For full functionality on Linux GPU machines later: NVIDIA drivers / CUDA / PyTorch (not required for the current environment checks)

---

### Quickstart (recommended: clean virtualenv)

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate            # macOS/Linux
# .\.venv\Scripts\activate           # Windows PowerShell

python -m pip install -U pip
python -m pip install -e .

which continuum
continuum --help
continuum doctor
```
![Status](https://img.shields.io/badge/status-early--development-orange)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
