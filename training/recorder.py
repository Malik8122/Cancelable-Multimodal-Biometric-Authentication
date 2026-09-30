"""Standard training record for every modality (config, environment, epoch log, checkpoints, figures).

Layout of one run (`training/<modality>/runs/<experiment_id>/`):

    config.json                  - what the code was configured to do (values read from code; unknown -> NOT_AVAILABLE)
    environment.json             - measured at run start from the live runtime
    training_log.csv             - one row per epoch, appended and flushed after EVERY epoch (crash-safe)
    checkpoints/                 - new checkpoints (gitignored) + checkpoint_manifest.json
    best_checkpoint.json         - explicit best-checkpoint rule and result
    metrics/                     - extra measured metrics (e.g. final evaluation)
    figures/                     - per-metric curves, generated only for columns that contain values

`training/<modality>/latest` is a text file naming the most recent run.
Nothing here writes to the deployed checkpoints under models/*/saved/.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NA = "NOT_AVAILABLE"

LOG_COLUMNS = [
    "epoch", "timestamp_utc", "learning_rate", "train_loss", "validation_loss", "train_accuracy", "validation_accuracy",
    "validation_EER", "validation_AUC", "elapsed_seconds", "cumulative_training_seconds",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip() or NA
    except Exception:  # noqa: BLE001
        return NA


def environment() -> dict:
    """Measured from the running process - never typed by hand."""
    import importlib

    env = {"timestamp_utc": _now(), "os": f"{platform.system()} {platform.release()} ({platform.version()})",
           "python": platform.python_version(), "cpu": platform.processor() or NA, "cpu_logical_cores": os.cpu_count(),
           "git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
           "git_worktree_dirty": bool(git("status", "--porcelain") not in ("", NA))}
    try:
        name = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                              capture_output=True, text=True, timeout=30).stdout.strip() if os.name == "nt" else ""
        if name:
            env["cpu"] = name
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil

        env["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except Exception:  # noqa: BLE001
        env["ram_gb"] = NA
    try:
        import torch

        env["pytorch"] = torch.__version__
        env["cuda"] = torch.version.cuda or NA
        env["cuda_available"] = torch.cuda.is_available()
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        env["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else NA
        env["torch_threads"] = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        env.update({"pytorch": NA, "cuda": NA, "gpu": NA})
    packages = {}
    for mod in ("numpy", "scipy", "sklearn", "torchvision", "torchaudio", "speechbrain", "facenet_pytorch", "cv2"):
        try:
            m = importlib.import_module(mod)
            packages[mod] = getattr(m, "__version__", "installed")
        except Exception:  # noqa: BLE001
            packages[mod] = NA
    env["packages"] = packages
    return env


def _clean(v):
    if v is None:
        return NA
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return v.item()
    except Exception:  # noqa: BLE001
        pass
    return v


class TrainingRecorder:
    def __init__(self, modality: str, config: dict, experiment_id: str | None = None, root: Path | None = None):
        self.modality = modality
        self.experiment_id = experiment_id or f"{modality}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        self.dir = (root or REPO / "training" / modality / "runs") / self.experiment_id
        for sub in ("checkpoints", "metrics", "figures"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        config = {"experiment_id": self.experiment_id, "timestamp_utc": _now(), "modality": modality, **config,
                  "git_commit": git("rev-parse", "HEAD")}
        (self.dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
        (self.dir / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n", encoding="utf-8")
        (self.dir.parent.parent / "latest").write_text(self.experiment_id + "\n", encoding="utf-8")
        self.log_path = self.dir / "training_log.csv"
        with self.log_path.open("w", newline="", encoding="utf-8") as h:
            csv.DictWriter(h, fieldnames=LOG_COLUMNS).writeheader()
        self.manifest_path = self.dir / "checkpoints" / "checkpoint_manifest.json"
        self.manifest: list[dict] = []
        self.seed = config.get("random_seed", NA)
        self.start = time.perf_counter()
        self.epoch_start = self.start
        self.cumulative = 0.0

    def start_epoch(self):
        self.epoch_start = time.perf_counter()

    def log_epoch(self, epoch: int, **metrics) -> dict:
        """Append one row and flush immediately (a crash keeps every completed epoch)."""
        elapsed = time.perf_counter() - self.epoch_start
        self.cumulative += elapsed
        row = {c: NA for c in LOG_COLUMNS}
        row.update({"epoch": epoch, "timestamp_utc": _now(), "elapsed_seconds": round(elapsed, 2),
                    "cumulative_training_seconds": round(self.cumulative, 2)})
        for k, v in metrics.items():
            if k not in LOG_COLUMNS:
                raise KeyError(f"unknown log column {k!r}")
            row[k] = _clean(v)
        with self.log_path.open("a", newline="", encoding="utf-8") as h:
            csv.DictWriter(h, fieldnames=LOG_COLUMNS).writerow(row)
            h.flush()
            os.fsync(h.fileno())
        self.epoch_start = time.perf_counter()  # next epoch's clock starts when this one is recorded
        print(f"[{self.modality}] " + " ".join(f"{k}={row[k]}" for k in LOG_COLUMNS if row[k] != NA), flush=True)
        return row

    def record_checkpoint(self, filename: str, epoch: int, **metrics):
        entry = {"checkpoint": filename, "epoch": epoch, "timestamp_utc": _now(), "git_commit": git("rev-parse", "HEAD"),
                 "random_seed": self.seed, **{k: _clean(v) for k, v in metrics.items()}}
        self.manifest = [m for m in self.manifest if m["checkpoint"] != filename] + [entry]
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")

    def record_best(self, best_epoch, best_metric, metric_name: str, checkpoint_path: str, rule: str):
        (self.dir / "best_checkpoint.json").write_text(json.dumps(
            {"best_epoch": best_epoch, "best_metric": _clean(best_metric), "metric_name": metric_name,
             "checkpoint_path": checkpoint_path, "selection_rule": rule}, indent=2) + "\n", encoding="utf-8")

    def save_metrics(self, name: str, obj: dict):
        (self.dir / "metrics" / f"{name}.json").write_text(json.dumps(obj, indent=2, default=_clean) + "\n", encoding="utf-8")

    def finish(self, status: str):
        """Figures for every metric column that actually has values, plus a run status file."""
        (self.dir / "run_status.json").write_text(json.dumps({"status": status, "finished_utc": _now(),
                                                              "total_seconds": round(time.perf_counter() - self.start, 1)}, indent=2) + "\n",
                                                  encoding="utf-8")
        plot_run(self.dir)


def plot_run(run_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (run_dir / "training_log.csv").open(encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    if not rows:
        return

    def series(col):
        pts = [(int(r["epoch"]), float(r[col])) for r in rows if r[col] not in (NA, "")]
        return pts

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})
    groups = [("loss", ["train_loss", "validation_loss"], "loss"), ("accuracy", ["train_accuracy", "validation_accuracy"], "accuracy"),
              ("validation_eer", ["validation_EER"], "EER"), ("validation_auc", ["validation_AUC"], "ROC-AUC")]
    for name, cols, ylabel in groups:
        present = [(c, series(c)) for c in cols if series(c)]
        if not present:
            continue
        fig, ax = plt.subplots(figsize=(3.5, 2.3))
        for c, pts in present:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=2.5, lw=1, label=c)
        ax.set(xlabel="epoch", ylabel=ylabel, title=f"{run_dir.parent.parent.name}: {ylabel} vs epoch")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=6)
        fig.savefig(run_dir / "figures" / f"{name}_vs_epoch.png")
        plt.close(fig)
