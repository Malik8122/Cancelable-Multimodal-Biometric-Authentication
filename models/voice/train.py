"""Voice training loop: AdamW + CosineAnnealingLR + AMP + early stopping.

An importable script (`if __name__ == "__main__":` at the bottom), not
notebook-only code - notebooks/04_voice_training.ipynb and
kaggle_kernels/voice_training/voice-embedding-training.ipynb both import and
call `train()` rather than duplicating the training loop inline, per the
spec's "avoid notebook-only code inside training modules."

Requires `speechbrain` (for the real ECAPA-TDNN backbone) and a VoxCeleb1
directory - not runnable on this dev machine (see docs/VOICE_MODEL.md); it
is exercised on Kaggle/Colab GPU runtimes, the same way
notebooks/01-03_*_training_and_testing.ipynb's training cells are.
"""

from __future__ import annotations

import logging
from pathlib import Path

from models.common.checkpoint_io import save_state_dict_as_h5
from models.voice.config import VoiceConfig
from models.voice.dataset import VoxCelebDataset
from models.voice.losses import build_arcface_head
from models.voice.model import VoiceEmbeddingNet
from models.voice.utils import detect_device

logger = logging.getLogger("models.voice.train")


def train(
    dataset_root: str | Path,
    output_dir: str | Path,
    config: VoiceConfig | None = None,
    device: str | None = None,
    epoch_callback=None,
) -> Path:
    """Fine-tune the voice embedding backbone; returns the best checkpoint's path.

    Saves `voice_embedder_best.pt`/`voice_embedder_last.pt` during training,
    then the canonical `voice_embedder.pt` (+ `.h5`) at the end.

    `device` defaults to `detect_device()` (which itself smoke-tests CUDA,
    not just checks availability) but can be overridden explicitly - e.g. by
    a caller that already ran its own device check and wants to reuse it
    rather than re-running the smoke test.

    `epoch_callback(info)` (optional, observation only) is called after every epoch's checkpointing with
    `epoch`, `train_loss`, `validation_loss`, `learning_rate`, `saved_best`, `model`, `val_dataset` - it cannot change
    training (used by training/run_training.py to write the standard training record).
    """
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    config = config or VoiceConfig()
    device = device or detect_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = VoxCelebDataset(dataset_root, mode="train", config=config)
    val_dataset = VoxCelebDataset(dataset_root, mode="val", config=config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = VoiceEmbeddingNet(config).to(device)
    arcface_head = build_arcface_head(train_dataset.num_speakers, config).to(device)

    optimizer = AdamW(
        list(model.parameters()) + list(arcface_head.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    scaler = torch.amp.GradScaler(device, enabled=config.mixed_precision and device == "cuda")
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_checkpoint_path = output_dir / "voice_embedder_best.pt"
    last_checkpoint_path = output_dir / "voice_embedder_last.pt"

    for epoch in range(config.num_epochs):
        model.train()
        train_loss_sum, train_count = 0.0, 0
        lr_this_epoch = optimizer.param_groups[0]["lr"]
        for mel_batch, label_batch in train_loader:
            mel_batch = mel_batch.transpose(1, 2).to(device)  # (batch, n_mels, T) -> (batch, T, n_mels)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device, enabled=config.mixed_precision and device == "cuda"):
                embeddings = model(mel_batch)
                logits = arcface_head(embeddings, label_batch)
                loss = criterion(logits, label_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * label_batch.size(0)  # logging only
            train_count += label_batch.size(0)
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for mel_batch, label_batch in val_loader:
                mel_batch = mel_batch.transpose(1, 2).to(device)
                label_batch = label_batch.to(device)
                embeddings = model(mel_batch)
                logits = arcface_head(embeddings, label_batch)
                val_losses.append(criterion(logits, label_batch).item())
        val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")
        logger.info("epoch=%d val_loss=%.4f", epoch, val_loss)

        torch.save(model.state_dict(), last_checkpoint_path)
        saved_best = val_loss < best_val_loss
        stop = False
        if saved_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_checkpoint_path)
        else:
            epochs_without_improvement += 1
            stop = epochs_without_improvement >= config.early_stopping_patience
        if epoch_callback is not None:
            epoch_callback({"epoch": epoch, "train_loss": train_loss_sum / max(train_count, 1), "validation_loss": val_loss,
                            "learning_rate": lr_this_epoch, "saved_best": saved_best, "model": model, "val_dataset": val_dataset,
                            "early_stop": stop})
        if stop:
            logger.info("Early stopping at epoch=%d (best_val_loss=%.4f)", epoch, best_val_loss)
            break

    final_state_dict = torch.load(best_checkpoint_path, map_location=device)
    canonical_pt_path = output_dir / "voice_embedder.pt"
    canonical_h5_path = output_dir / "voice_embedder.h5"
    torch.save(final_state_dict, canonical_pt_path)
    save_state_dict_as_h5(final_state_dict, canonical_h5_path)
    config.to_json(output_dir / "training_config.json")

    return canonical_pt_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Fine-tune the Voice (ECAPA-TDNN) embedding model.")
    parser.add_argument("--dataset-root", required=True, help="VoxCeleb1 root directory (e.g. /kaggle/input/<slug>).")
    parser.add_argument("--output-dir", default="models/voice/saved", help="Where checkpoints are written.")
    args = parser.parse_args()

    train(args.dataset_root, args.output_dir)
