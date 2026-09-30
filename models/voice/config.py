"""All Voice module hyperparameters in one place.

Every audio-pipeline and training knob is a field here rather than scattered
constants, per the spec's "everything configurable inside config.py" —
unlike Face (which hardcodes a handful of module-level
constants directly in inference.py), Voice has enough independently-tunable
parameters (sample rate, clip length, mel bins, FFT/hop, VAD backend, plus a
full training recipe) that a single config object is worth the extra
structure. Defaults match docs/VOICE_MODEL.md's hyperparameter table.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass
class VoiceConfig:
    # --- Audio pipeline (preprocessing/voice.py) ---
    sample_rate: int = 16_000
    clip_seconds: float = 4.0
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    vad_backend: Literal["energy", "webrtcvad"] = "energy"
    vad_energy_threshold_ratio: float = 0.02
    target_rms: float = 0.1

    # --- Augmentation (training only - preprocessing/voice.py never applies these) ---
    augmentation_enabled: bool = False
    gaussian_noise_std: float = 0.005
    speed_perturb_rates: tuple[float, ...] = (0.9, 1.0, 1.1)
    time_mask_max_frames: int = 10
    freq_mask_max_bins: int = 8
    random_gain_db_range: tuple[float, float] = (-6.0, 6.0)

    # --- Model ---
    embedding_dim: int = 192

    # --- ArcFace head (models/voice/losses.py) ---
    arcface_margin: float = 0.50
    arcface_scale: float = 30.0

    # --- Training (models/voice/train.py) ---
    batch_size: int = 64
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    mixed_precision: bool = True
    early_stopping_patience: int = 5
    train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15)

    #: Fields declared as tuples above - JSON has no tuple type, so
    #: `from_json` converts these back from the lists `json.loads` produces
    #: rather than leaving them as lists (which would silently change the
    #: field's runtime type after a save/load round-trip).
    _TUPLE_FIELDS = ("speed_perturb_rates", "random_gain_db_range", "train_val_test_split")

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "VoiceConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for field_name in cls._TUPLE_FIELDS:
            if field_name in raw:
                raw[field_name] = tuple(raw[field_name])
        return cls(**raw)
