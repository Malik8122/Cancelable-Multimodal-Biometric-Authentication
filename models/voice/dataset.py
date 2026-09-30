"""VoxCeleb1-shaped speaker dataset.

No hardcoded Kaggle path - `root_dir` is a constructor argument, so this
works identically against a real Kaggle-mounted VoxCeleb1 directory
(`/kaggle/input/<dataset-slug>`) and against a small synthetic directory in a
test's `tmp_path` (see tests/test_voice_dataset.py). Expects the common
VoxCeleb1 layout: `root_dir/<speaker_id>/<video_id>/<utterance>.wav` (only
the top-level `<speaker_id>` directory name is actually relied on - the
utterance/video nesting below it is walked generically), but tolerates any
depth of subdirectories between `root_dir` and the `.wav` files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from models.voice.config import VoiceConfig
from preprocessing.voice import VoicePreprocessor, load_wav_file


def _discover_utterances(root_dir: Path) -> list[tuple[Path, str]]:
    """Walk `root_dir`, returning (wav_path, speaker_id) pairs.

    `speaker_id` is the path component directly under `root_dir` - VoxCeleb1
    speaker IDs (e.g. `id10001`) are exactly this, and this also degrades
    gracefully for a flatter synthetic test layout
    (`root_dir/<speaker_id>/<file>.wav`).
    """
    utterances = []
    for wav_path in sorted(root_dir.rglob("*.wav")):
        speaker_id = wav_path.relative_to(root_dir).parts[0]
        utterances.append((wav_path, speaker_id))
    return utterances


def _split_indices(
    num_samples: int, split: tuple[float, float, float], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    train_fraction, val_fraction, _ = split
    n_train = max(1, int(num_samples * train_fraction))
    n_val = max(1, int(num_samples * val_fraction)) if num_samples - n_train > 1 else 0
    return indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]


class VoxCelebDataset:
    """PyTorch-`Dataset`-compatible speaker-verification dataset.

    Deliberately does not import `torch` at module load time - only
    `__getitem__` converts the preprocessed mel-spectrogram to a tensor - so
    this class (and its tests) work even before `torch` is confirmed
    available, matching every other lazy-import boundary in this project.
    Per-identity 70/15/15 split (configurable via `config.train_val_test_split`),
    a deliberate simplification instead of VoxCeleb1's official trial-list protocol - see
    docs/VOICE_MODEL.md for why.
    """

    def __init__(
        self,
        root_dir: str | Path,
        mode: Literal["train", "val", "test"] = "train",
        config: VoiceConfig | None = None,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.config = config or VoiceConfig()
        self.preprocessor = VoicePreprocessor(
            sample_rate=self.config.sample_rate,
            clip_seconds=self.config.clip_seconds,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )

        utterances = _discover_utterances(self.root_dir)
        if not utterances:
            raise ValueError(f"No .wav files found under {self.root_dir} - check the dataset path.")

        speaker_ids = sorted({speaker_id for _, speaker_id in utterances})
        self.speaker_to_label = {speaker_id: label for label, speaker_id in enumerate(speaker_ids)}
        self.num_speakers = len(speaker_ids)

        rng = np.random.default_rng(seed)
        by_speaker: dict[str, list[int]] = {speaker_id: [] for speaker_id in speaker_ids}
        for index, (_, speaker_id) in enumerate(utterances):
            by_speaker[speaker_id].append(index)

        selected_indices: list[int] = []
        for indices in by_speaker.values():
            train_idx, val_idx, test_idx = _split_indices(len(indices), self.config.train_val_test_split, rng)
            split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
            selected_indices.extend(np.array(indices)[split_map[mode]].tolist())

        self._items = [utterances[i] for i in selected_indices]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        import torch

        wav_path, speaker_id = self._items[index]
        waveform, sample_rate = load_wav_file(wav_path)
        # `rng=None` lets VoicePreprocessor draw fresh randomness per call in
        # training mode (a different random crop each epoch is the point of
        # this augmentation); non-training mode ignores `rng` entirely since
        # center-crop is deterministic (see VoicePreprocessor._fixed_length_segment).
        mel = self.preprocessor.preprocess(
            waveform,
            sample_rate=sample_rate,
            training=(self.mode == "train"),
            rng=None,
        )
        return torch.from_numpy(mel).float(), self.speaker_to_label[speaker_id]
