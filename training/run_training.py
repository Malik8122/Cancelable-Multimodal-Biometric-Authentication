"""Reproducible training runs for face and voice, with the standard training record.

    python -m training.run_training face
    python -m training.run_training voice        [--stop-after-epochs N]

- face: re-implements kaggle_kernels/face_training/face-embedding-training.ipynb cell by cell (the notebook is the
  only face training code in the repository). Differences, all documented in config.json: torch/numpy seeds are set
  (the notebook set none), and validation loss / EER / AUC are additionally logged (the notebook logged val_acc).
- voice: calls the repository's own training loop (models/voice/train.py) unchanged, observing it through its
  `epoch_callback` hook.
- Checkpoints go ONLY to training/<modality>/runs/<id>/checkpoints/ (gitignored); models/*/saved/ is never touched.
- `--stop-after-epochs N` ends a run after N completed epochs without altering its configuration or LR schedule;
  the run is then recorded as STOPPED (partial), never as a completed training.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

from training.recorder import NA, REPO, TrainingRecorder

DATA = REPO / "data"
SEED = 42


class StopTraining(Exception):
    pass


def _seed(seed: int = SEED):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _verification(embeddings: np.ndarray, labels) -> dict:
    """All-pairs cosine verification (the protocol of evaluation/experiments.py), vectorized."""
    from evaluation.ieee.common import auc, eer, roc

    E = np.asarray(embeddings, float)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    labels = np.asarray(labels)
    iu = np.triu_indices(len(E), 1)
    s = (E @ E.T)[iu]
    same = (labels[:, None] == labels[None, :])[iu]
    e, t = eer(s[same], s[~same])
    return {"EER": e, "EER_threshold": t, "AUC": auc(roc(s[same], s[~same])), "genuine_pairs": int(same.sum()),
            "impostor_pairs": int((~same).sum()), "samples": len(E)}


# ----------------------------------------------------------------------------- face


def train_face(stop_after: int | None = None, alignment: str = "bbox", source: str = "sklearn_slice", experiment_id: str | None = None) -> Path:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from facenet_pytorch import InceptionResnetV1
    from sklearn.datasets import fetch_lfw_people
    from torch.utils.data import DataLoader, Dataset

    from models.common.arcface import ArcMarginProduct
    from models.face.inference import FACE_EMBEDDING_DIM
    from preprocessing.face import ALIGNMENT_TEMPLATE_160, AlignmentFailed, FacePreprocessor, FacePreprocessorAligned

    _seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if source == "full_frame":
        # The SAME identities/files as fetch_lfw_people(min_faces_per_person=20) (persons sorted, files sorted), but the full
        # 250x250 funneled frames instead of sklearn's default 125x94 slice: the slice cuts off the context an aligned crop
        # needs (measured: aligned crops of sliced images are ~39% black border vs ~1% for full frames).
        import cv2

        root = DATA / "lfw" / "lfw_home" / "lfw_funneled"
        # replicate sklearn.datasets._lfw._fetch_lfw_people exactly: sorted folders/files, names with spaces,
        # np.unique targets, then the RandomState(42) shuffle
        person_names, files = [], []
        for d in sorted(root.iterdir()):
            fs = sorted(d.glob("*.jpg")) if d.is_dir() else []
            if len(fs) >= 20:
                person_names += [d.name.replace("_", " ")] * len(fs)
                files += fs
        names = np.unique(person_names)
        labels = np.searchsorted(names, person_names)
        order = np.arange(len(files))
        np.random.RandomState(42).shuffle(order)
        labels = labels[order]
        images = np.stack([cv2.cvtColor(cv2.imread(str(files[k])), cv2.COLOR_BGR2RGB) for k in order])
        ref = fetch_lfw_people(min_faces_per_person=20, resize=0.1, color=False, funneled=True, data_home=str(DATA / "lfw"))
        assert np.array_equal(names, ref.target_names) and np.array_equal(labels, ref.target), "full-frame set differs from the sklearn set"
    else:
        lfw = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True, funneled=True, data_home=str(DATA / "lfw"))  # cell 4
        images = (lfw.images * 255).astype(np.uint8)
        labels, names = lfw.target, lfw.target_names
    pre = FacePreprocessorAligned(device=device) if alignment == "similarity" else FacePreprocessor(device=device)  # cell 6
    aligned, aligned_labels, skipped, alignment_failed, kept = [], [], 0, 0, []
    for i, (img, label) in enumerate(zip(images, labels)):
        try:
            aligned.append(pre.preprocess(img))
            aligned_labels.append(label)
            kept.append(i)
        except AlignmentFailed:
            alignment_failed += 1
        except ValueError:
            skipped += 1
    aligned, aligned_labels = np.stack(aligned), np.array(aligned_labels)
    rng = np.random.default_rng(42)  # cell 8
    by_identity: dict = {}
    for idx, label in enumerate(aligned_labels):
        by_identity.setdefault(label, []).append(idx)
    train_idx, val_idx, test_idx = [], [], []
    for _, idxs in by_identity.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train, n_val = max(1, int(n * 0.7)), max(1, int(n * 0.15))
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:] if n - n_train - n_val > 0 else idxs[-1:])

    config = {
        "model": "InceptionResnetV1 (facenet-pytorch), pretrained='vggface2'; trainable: block8, last_linear, last_bn + ArcFace head",
        "embedding_dimension": FACE_EMBEDDING_DIM, "dataset": "LFW funneled via sklearn.datasets.fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True)",
        "dataset_version": (f"LFW funneled full frames {images.shape[1]}x{images.shape[2]} (same files as the sklearn set)" if source == "full_frame"
                            else f"scikit-learn LFW (default slice -> {images.shape[1]}x{images.shape[2]} crops)"),
        "image_source": source,
        "num_subjects": int(len(names)), "num_samples": int(len(images)), "samples_after_face_detection": int(len(aligned)),
        "skipped_no_face": skipped, "skipped_alignment_failed": alignment_failed,
        "kept_image_indices_sha256": __import__("hashlib").sha256(np.array(kept, dtype=np.int64).tobytes()).hexdigest(),
        "split": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx),
                  "rule": "per-identity 70/15/15, np.random.default_rng(42) (notebook cell 8); identities shared across splits (closed-set)"},
        "preprocessing": {"detector": "MTCNN image_size=160, margin=0, post_process=True",
                          "alignment": ("5-landmark similarity transform (Umeyama, no reflection) to preprocessing/face.py::ALIGNMENT_TEMPLATE_160 "
                                        f"{ALIGNMENT_TEMPLATE_160.tolist()}, bilinear, constant black border" if alignment == "similarity"
                                        else "bounding-box crop (landmarks used only for quality gates in enrollment, not here)"),
                          "face_selection": "largest detected face", "input_normalization": "(x - 127.5) / 128"},
        "augmentation": "none", "loss": "ArcFace (ArcMarginProduct s=30.0, m=0.50) + CrossEntropy", "optimizer": "Adam",
        "learning_rate": 1e-4, "scheduler": "none", "batch_size": 32, "configured_epochs": 10, "random_seed": SEED,
        "checkpoint_selection": "final epoch (the notebook saves the model after the last epoch; there is no best-checkpoint selection)",
        "differences_from_historical_notebook": ["torch/numpy/random seeded with 42 (notebook unseeded)",
                                                 "validation loss, EER and AUC additionally logged (notebook logged val_acc only)"],
        "source": "kaggle_kernels/face_training/face-embedding-training.ipynb cells 4-16",
    }
    if source == "full_frame":
        config["differences_from_historical_notebook"].append("full-frame 250x250 LFW images instead of the sklearn 125x94 slice (see image_source)")
    if experiment_id:
        config["controlled_variable"] = f"face preprocessing ({alignment}) - all else identical within the pair"
        folder = {"face_aligned_v1": "aligned_v1", "face_aligned_v2": "aligned_v2", "face_bbox_fullframe_v2": "bbox_fullframe_v2"}[experiment_id]
        rec = TrainingRecorder("face", config, experiment_id=experiment_id, root=REPO / "training" / "face" / folder / "runs")
    elif alignment == "similarity":
        config["controlled_variable"] = "face preprocessing (aligned) - everything else identical to the bbox baseline run of this runner"
        rec = TrainingRecorder("face", config, experiment_id="face_aligned_v1", root=REPO / "training" / "face" / "aligned_v1" / "runs")
    else:
        rec = TrainingRecorder("face", config)

    class FaceDataset(Dataset):
        def __init__(self, imgs, labs):
            self.imgs, self.labs = imgs, labs

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, i):
            t = torch.from_numpy(self.imgs[i]).permute(2, 0, 1).float()
            return (t - 127.5) / 128.0, int(self.labs[i])

    backbone = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)  # cell 10
    head = ArcMarginProduct(FACE_EMBEDDING_DIM, len(names)).to(device)
    for n_, p in backbone.named_parameters():
        p.requires_grad = any(n_.startswith(x) for x in ("block8", "last_linear", "last_bn"))
    trainable = [p for p in backbone.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = optim.Adam(trainable, lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(FaceDataset(aligned[train_idx], aligned_labels[train_idx]), batch_size=32, shuffle=True)
    val_loader = DataLoader(FaceDataset(aligned[val_idx], aligned_labels[val_idx]), batch_size=32)
    ckpt = rec.dir / "checkpoints" / "face_embedder.pt"
    status = "COMPLETED"
    for epoch in range(10):  # cell 12
        rec.start_epoch()
        backbone.train()
        tl, tc, tt = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = head(backbone(x), y)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tl += loss.item() * x.size(0)
            tc += (logits.argmax(1) == y).sum().item()
            tt += x.size(0)
        backbone.eval()
        vl, vc, vt, vemb, vlab = 0.0, 0, 0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                emb = backbone(x)
                logits = head(emb, y)
                vl += criterion(logits, y).item() * x.size(0)
                vc += (logits.argmax(1) == y).sum().item()
                vt += x.size(0)
                vemb.append(emb.cpu().numpy())
                vlab += y.cpu().tolist()
        ver = _verification(np.vstack(vemb), vlab)
        rec.log_epoch(epoch + 1, learning_rate=optimizer.param_groups[0]["lr"], train_loss=tl / tt, validation_loss=vl / vt,
                      train_accuracy=tc / tt, validation_accuracy=vc / vt, validation_EER=ver["EER"], validation_AUC=ver["AUC"])
        if stop_after and epoch + 1 >= stop_after and epoch + 1 < 10:
            status = f"STOPPED after {epoch + 1} of 10 epochs (--stop-after-epochs)"
            break
    backbone.eval()
    torch.save(backbone.state_dict(), ckpt)  # cell 14
    rec.record_checkpoint(ckpt.name, epoch + 1, train_loss=tl / tt, validation_loss=vl / vt, validation_accuracy=vc / vt,
                          validation_EER=ver["EER"], validation_AUC=ver["AUC"])
    rec.record_best(epoch + 1, vc / vt, "validation_accuracy (final epoch)", str(ckpt.relative_to(REPO)),
                    "final epoch - the historical notebook has no best-checkpoint selection")
    from models.face.inference import FaceEmbedder  # cell 16: test evaluation of the saved checkpoint

    emb = FaceEmbedder(checkpoint_path=ckpt, device=device)
    test = _verification(np.vstack([emb.extract_embedding(aligned[i]) for i in test_idx]), [names[aligned_labels[i]] for i in test_idx])
    rec.save_metrics("test_verification", {**test, "note": "test identities are the training identities (closed-set, as in the notebook)"})
    rec.finish(status)
    return rec.dir


# ----------------------------------------------------------------------------- voice


def train_voice(stop_after: int | None = None) -> Path:
    import torch

    from models.voice.config import VoiceConfig
    from models.voice.dataset import VoxCelebDataset
    from models.voice.train import train

    _seed()
    root = DATA / "voxceleb_subset" / "vox1_indian" / "content" / "vox_indian"
    cfg = VoiceConfig()
    splits = {m: VoxCelebDataset(root, mode=m, config=cfg) for m in ("train", "val", "test")}
    config = {
        "model": "SpeechBrain ECAPA_TDNN(input_size=80, lin_neurons=192), trained from scratch (no pretrained speaker weights)",
        "embedding_dimension": cfg.embedding_dim, "dataset": "Kaggle gaurav41/voxceleb1-audio-wav-files-for-india-celebrity",
        "dataset_version": "Kaggle download (version as served at download time; version id NOT_AVAILABLE)",
        "num_subjects": splits["train"].num_speakers, "num_samples": sum(len(d) for d in splits.values()),
        "split": {"train": len(splits["train"]), "validation": len(splits["val"]), "test": len(splits["test"]),
                  "rule": "per-speaker 70/15/15, seed 42 (models/voice/dataset.py); speakers shared across splits (closed-set)"},
        "preprocessing": {k: getattr(cfg, k) for k in ("sample_rate", "clip_seconds", "n_mels", "n_fft", "hop_length", "vad_backend",
                                                       "vad_energy_threshold_ratio", "target_rms")},
        "augmentation": {"enabled": cfg.augmentation_enabled, "configured_options_unused": {k: getattr(cfg, k) for k in (
            "gaussian_noise_std", "speed_perturb_rates", "time_mask_max_frames", "freq_mask_max_bins", "random_gain_db_range")}},
        "loss": f"ArcFace (margin {cfg.arcface_margin}, scale {cfg.arcface_scale}) + CrossEntropy",
        "optimizer": f"AdamW (weight_decay {cfg.weight_decay})", "learning_rate": cfg.learning_rate,
        "scheduler": f"CosineAnnealingLR(T_max={cfg.num_epochs})", "batch_size": cfg.batch_size, "configured_epochs": cfg.num_epochs,
        "early_stopping_patience": cfg.early_stopping_patience, "mixed_precision": f"{cfg.mixed_precision} (effective only on CUDA)",
        "random_seed": SEED, "checkpoint_selection": "minimum validation loss (models/voice/train.py); validation EER/AUC are logged by this runner but NOT used for selection",
        "differences_from_historical_run": ["torch/numpy/random seeded with 42", "train loss and validation EER/AUC additionally logged"],
        "source": "models/voice/train.py, models/voice/config.py",
    }
    rec = TrainingRecorder("voice", config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckdir = rec.dir / "checkpoints"
    best = {"epoch": None, "val_loss": None}

    def val_embeddings(model, dataset):
        model.eval()
        embs, labs = [], []
        with torch.no_grad():
            for i in range(len(dataset)):
                mel, lab = dataset[i]
                embs.append(model(mel.unsqueeze(0).transpose(1, 2).to(device)).cpu().numpy()[0])
                labs.append(int(lab))
        return np.array(embs), labs

    def cb(info):
        e, l = val_embeddings(info["model"], info["val_dataset"])
        ver = _verification(e, l)
        ep = info["epoch"] + 1
        rec.log_epoch(ep, learning_rate=info["learning_rate"], train_loss=info["train_loss"], validation_loss=info["validation_loss"],
                      validation_EER=ver["EER"], validation_AUC=ver["AUC"])
        rec.record_checkpoint("voice_embedder_last.pt", ep, train_loss=info["train_loss"], validation_loss=info["validation_loss"],
                              validation_EER=ver["EER"], validation_AUC=ver["AUC"])
        if info["saved_best"]:
            best.update(epoch=ep, val_loss=info["validation_loss"])
            rec.record_checkpoint("voice_embedder_best.pt", ep, train_loss=info["train_loss"], validation_loss=info["validation_loss"],
                                  validation_EER=ver["EER"], validation_AUC=ver["AUC"])
        if stop_after and ep >= stop_after and not info["early_stop"]:
            raise StopTraining

    status = "COMPLETED"
    try:
        train(root, ckdir, config=cfg, device=device, epoch_callback=cb)
    except StopTraining:
        status = f"STOPPED after {stop_after} of {cfg.num_epochs} epochs (--stop-after-epochs); canonical checkpoint not produced"
    rec.record_best(best["epoch"], best["val_loss"], "validation_loss (minimum)", str((ckdir / "voice_embedder_best.pt").relative_to(REPO)),
                    "minimum validation loss, as implemented in models/voice/train.py")
    if best["epoch"] is not None:
        from models.voice.model import VoiceEmbeddingNet

        m = VoiceEmbeddingNet(cfg).to(device)
        m.load_state_dict(torch.load(ckdir / "voice_embedder_best.pt", map_location=device))
        e, l = val_embeddings(m, splits["test"])
        rec.save_metrics("test_verification", {**_verification(e, l), "checkpoint": "voice_embedder_best.pt",
                                               "note": "closed-set: test speakers also appear in training"})
    rec.finish(status)
    return rec.dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("modality", choices=["face", "voice"])
    ap.add_argument("--stop-after-epochs", type=int, default=None)
    ap.add_argument("--face-alignment", choices=["bbox", "similarity"], default="bbox", help="face only: preprocessing variant")
    ap.add_argument("--face-source", choices=["sklearn_slice", "full_frame"], default="sklearn_slice", help="face only: LFW image source")
    ap.add_argument("--experiment-id", default=None, help="face only: face_aligned_v1 | face_aligned_v2 | face_bbox_fullframe_v2")
    a = ap.parse_args()
    t = time.perf_counter()
    if a.modality == "face":
        out = train_face(a.stop_after_epochs, alignment=a.face_alignment, source=a.face_source, experiment_id=a.experiment_id)
    else:
        out = train_voice(a.stop_after_epochs)
    print(f"done in {time.perf_counter() - t:.0f}s -> {out}")
    sys.exit(0)
