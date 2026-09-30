"""Offline tests for models/voice/utils.py."""

from __future__ import annotations

from models.voice.utils import detect_device


def test_detect_device_returns_a_valid_device_string():
    assert detect_device() in ("cpu", "cuda")


def test_detect_device_falls_back_to_cpu_when_a_cuda_op_raises(monkeypatch):
    """Regression test: a real Kaggle run hit CUDA reporting 'available' while
    the installed PyTorch build had no compiled kernel for that GPU's compute
    capability, which only surfaced as a crash deep inside training
    (AcceleratorError) rather than at this availability check. detect_device()
    must run a real op and fall back to CPU instead of trusting
    torch.cuda.is_available() alone."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _broken_zeros(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
        return torch.zeros(*args, **{k: v for k, v in kwargs.items() if k != "device"})

    monkeypatch.setattr(torch, "zeros", _broken_zeros)

    assert detect_device() == "cpu"
