from __future__ import annotations

import warnings

# Torch emits this when importing NVML bindings via the legacy module name.
# Keep CLI output clean while leaving unrelated warnings visible.
warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\. Please install nvidia-ml-py instead\.",
    category=FutureWarning,
)
