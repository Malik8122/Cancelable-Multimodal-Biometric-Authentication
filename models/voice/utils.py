"""Small shared helpers for the Voice module.

Config JSON load/save lives on `VoiceConfig` itself (config.py); this module
holds the one thing every other file needs and none of them should
re-derive independently: which device to run on.
"""

from __future__ import annotations


def detect_device() -> str:
    """"cuda" if a GPU is both reported available *and* actually usable, else "cpu".

    Deferred `torch` import, like every other heavy-dependency import in this
    project (see models/face/inference.py) - importing models/voice/utils.py
    alone must never require torch to be installed.

    `torch.cuda.is_available()` alone isn't sufficient: some hosted GPU
    sessions (observed on Kaggle's free-tier P100s) report CUDA as available
    while shipping a PyTorch build that doesn't include a compiled kernel for
    that GPU's (older) compute capability, which doesn't fail until the first
    real op runs, deep inside training - as `AcceleratorError: CUDA error: no
    kernel image is available for execution on the device`. Running a real op
    now and falling back to CPU on failure catches this upfront, for every
    caller (models/voice/train.py, models/voice/inference.py::load_model),
    rather than requiring each call site to remember its own smoke test.
    """
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    try:
        (torch.zeros(1, device="cuda") + 1).cpu()
    except Exception:
        return "cpu"
    return "cuda"
