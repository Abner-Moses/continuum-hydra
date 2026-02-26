# Continuum Setup State

This directory contains reproducibility artifacts from `continuum setup`.

## Files
- `env_manifest.json`: captured environment and install metadata
- `requirements.txt`: pinned package snapshot for setup-installed ML deps

## Snapshot
- python_version: 3.12.3
- platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- architecture: x86_64
- numpy: 2.4.2
- torch: 2.10.0+cu128
- torch_cuda_available: True

## Re-apply
`/home/rishi/continuum-hydra/.venv/bin/python3 -m pip install -r .continuum/state/requirements.txt`

## Manifest Path
.continuum/state/env_manifest.json
