"""Extract real embeddings with the SHIPPED pipelines (embeddings/pipelines.py) and cache them under data/eval_cache.

Evaluation sets (identities never used to fit the respective model wherever the repo's split allows it):
- face: LFW (funneled) identities with 2-19 images. The face model was fine-tuned on LFW identities with >= 20
  images (`min_faces_per_person=20`, docs/PROJECT_REPORT.md), so these identities are DISJOINT from fine-tuning.
- voice: the VoxCeleb1 Indian-celebrity subset TEST split reproduced with the training code
  (models/voice/dataset.py::VoxCelebDataset(mode="test", seed=42)). Utterances are unseen, but the split is
  per-speaker, so speakers overlap with training (closed-set) - reported as such.

Failures to acquire (e.g. MTCNN finds no face) are recorded, not dropped silently.
Run: python -m evaluation.ieee.extract_embeddings [face|voice ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from evaluation.ieee.common import CACHE, DATA

LFW_DIR = DATA / "lfw" / "lfw_home" / "lfw_funneled"
VOX_DIR = DATA / "voxceleb_subset" / "vox1_indian" / "content" / "vox_indian"


def _torch_threads():
    import torch

    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))


def _rgb(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"unreadable image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def face_items() -> list[tuple[Path, str]]:
    items = []
    for person in sorted(p for p in LFW_DIR.iterdir() if p.is_dir()):
        files = sorted(person.glob("*.jpg"))
        if 2 <= len(files) <= 19:
            items += [(f, person.name) for f in files]
    return items


def voice_items() -> list[tuple[Path, str]]:
    from models.voice.dataset import VoxCelebDataset

    ds = VoxCelebDataset(VOX_DIR, mode="test", seed=42)
    return [(Path(p), s) for p, s in ds._items]


def _run(name: str, items, embed, meta_fn):
    out = CACHE / f"{name}_embeddings.npz"
    if out.exists():
        print(f"{name}: cached ({out})")
        return
    embs, keep, failures = [], [], []
    start = time.perf_counter()
    for k, item in enumerate(items):
        try:
            embs.append(embed(item))
            keep.append(item)
        except Exception as error:  # noqa: BLE001 - a failure to acquire is a result, not a crash
            failures.append((str(item[0]), type(error).__name__, str(error)[:120]))
        if (k + 1) % 250 == 0:
            rate = (k + 1) / (time.perf_counter() - start)
            print(f"{name}: {k + 1}/{len(items)}  {rate:.1f}/s  failures={len(failures)}", flush=True)
    meta = meta_fn(keep)
    np.savez_compressed(
        out, embeddings=np.asarray(embs, dtype=np.float32), paths=np.array([str(i[0]) for i in keep]),
        failures=np.array(failures, dtype=object) if failures else np.zeros((0, 3), dtype=object),
        attempted=len(items), seconds=time.perf_counter() - start, **meta,
    )
    print(f"{name}: {len(keep)}/{len(items)} embedded, {len(failures)} failures, {time.perf_counter() - start:.0f}s -> {out}")


def extract_face():
    from embeddings.pipelines import FacePipeline

    pipeline = FacePipeline()
    assert not pipeline.is_mock, "face checkpoint missing (git lfs pull)"
    _run("face", face_items(), lambda it: pipeline.embed(_rgb(it[0])), lambda keep: {"labels": np.array([i[1] for i in keep])})


def extract_voice():
    from embeddings.pipelines import VoicePipeline
    from preprocessing.voice import load_wav_file

    pipeline = VoicePipeline()
    assert not pipeline.is_mock, "voice checkpoint missing"

    def embed(it):
        waveform, sr = load_wav_file(it[0])
        return pipeline.embed(waveform, sample_rate=sr)

    _run("voice", voice_items(), embed, lambda keep: {"labels": np.array([i[1] for i in keep])})


if __name__ == "__main__":
    _torch_threads()
    wanted = sys.argv[1:] or ["voice", "face"]
    for name in wanted:
        {"face": extract_face, "voice": extract_voice}[name]()
