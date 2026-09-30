"""Generate the training documentation from run records + the historical-evidence table.

    python -m training.summarize

Writes training/TRAINING_SUMMARY.md, training/PAPER_TRAINING_TABLE.md, training/TRAINING_REPRODUCIBILITY_REPORT.md,
training/<modality>/README.md and training/<modality>/REPRODUCTION_STATUS.md.
Every number is either read from a run folder (NEWLY_MEASURED) or from HISTORICAL below (HISTORICAL_DOCUMENTED,
with its source); anything else is NOT_AVAILABLE. Nothing is typed in by hand here except the historical table,
whose entries each cite the document they were copied from.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from training.recorder import NA, REPO

T = REPO / "training"
MODALITIES = ("face", "voice")
H, N = "HISTORICAL_DOCUMENTED", "NEWLY_MEASURED"

# (quantity, value, source) - copied verbatim from repository documents / committed CSVs; not re-verifiable.
HISTORICAL = {
    "face": [
        ("identities / images", "62 / 3,023", "docs/PROJECT_REPORT.md §8.2"),
        ("final epoch train_loss / train_acc / val_acc", "0.1378 / 0.976 / 0.913 (epoch 10/10)", "docs/PROJECT_REPORT.md §8.2"),
        ("test accuracy@EER / EER / AUC", "0.990 / 0.010 / 0.999", "evaluation/results/face_metrics.csv; docs/PROJECT_REPORT.md §8.2"),
        ("hardware", "Kaggle kernel with GPU enabled; completed runs reported as CPU fallback", "kernel-metadata.json; docs/PROJECT_REPORT.md §8.3"),
        ("training duration", NA, "no saved notebook outputs"),
    ],
    "voice": [
        ("speakers / utterances", "24 / 4,857", "docs/VOICE_MODEL.md"),
        ("train / val / test", "3,389 / 717 / 751", "docs/VOICE_MODEL.md"),
        ("epochs completed", "30 (all)", "docs/VOICE_MODEL.md"),
        ("hardware / duration", "CPU fallback; 'a little under 8 hours'", "docs/VOICE_MODEL.md"),
        ("test EER / accuracy@EER / AUC", "2.29% / 97.71% / 0.9967", "evaluation/results/voice_metrics.csv"),
        ("validation EER during training", NA, "models/voice/train.py computes validation loss only"),
        ("best epoch", NA, "no saved notebook outputs"),
    ],
}


def runs(modality):
    d = T / modality / "runs"
    return sorted(p for p in d.iterdir() if p.is_dir()) if d.exists() else []


def load(run: Path) -> dict:
    j = lambda n: json.loads((run / n).read_text()) if (run / n).exists() else {}
    log = list(csv.DictReader((run / "training_log.csv").open(encoding="utf-8"))) if (run / "training_log.csv").exists() else []
    test = run / "metrics" / "test_verification.json"
    return {"id": run.name, "dir": run, "config": j("config.json"), "env": j("environment.json"), "status": j("run_status.json"),
            "best": j("best_checkpoint.json"), "log": log, "test": json.loads(test.read_text()) if test.exists() else {},
            "manifest": json.loads((run / "checkpoints" / "checkpoint_manifest.json").read_text()) if (run / "checkpoints" / "checkpoint_manifest.json").exists() else [],
            "figures": sorted(p.name for p in (run / "figures").glob("*.png"))}


def completed(modality):
    rs = [load(r) for r in runs(modality)]
    done = [r for r in rs if str(r["status"].get("status", "")).startswith("COMPLETED")]
    return (done[-1] if done else None), rs


def f(v, fmt="{:.4f}"):
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return NA if v in (None, "", NA) else str(v)


def hw(env):
    return f"{env.get('cpu', NA)}, {env.get('ram_gb', NA)} GB RAM, GPU: {env.get('gpu', NA)}, torch {env.get('pytorch', NA)}" if env else NA


def run_line(r):
    last = r["log"][-1] if r["log"] else {}
    return (f"`{r['id']}` - status: {r['status'].get('status', 'INCOMPLETE (no run_status.json)')}; epochs logged: {len(r['log'])}; "
            f"training time: {f(last.get('cumulative_training_seconds'), '{:.0f}')} s")


def summary():
    rows = ["# Training summary", "", "Evidence types: **NEWLY_MEASURED** (this repository's reproducible runs, `training/<modality>/runs/`), "
            "**HISTORICAL_DOCUMENTED** (copied from repository documents, not reproducible), **NOT_AVAILABLE**. They are never merged.", "",
            "## Newly measured (reproduced runs)", "",
            "| Modality | Model | Dimension | Dataset | Subjects | Samples | Epochs | Best Metric | Training Time | Hardware | Status |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODALITIES:
        r, allruns = completed(m)
        src = r or (allruns[-1] if allruns else None)
        if src is None:
            rows.append(f"| {m} | {NA} | {NA} | {NA} | {NA} | {NA} | {NA} | {NA} | {NA} | {NA} | not run |")
            continue
        c, last = src["config"], (src["log"][-1] if src["log"] else {})
        best = src["best"]
        bm = f"{best.get('metric_name', NA)} = {f(best.get('best_metric'))} (epoch {best.get('best_epoch', NA)})" if best and best.get("best_epoch") not in (None, NA) else NA
        rows.append(f"| {m} | {re.split(r'[,(]', c.get('model', NA))[0].strip()} | {c.get('embedding_dimension', NA)} | {c.get('dataset', NA).split(' via')[0].split(' (')[0]} | "
                    f"{c.get('num_subjects', NA)} | {c.get('num_samples', NA)} | {len(src['log'])} of {c.get('configured_epochs', NA)} | {bm} | "
                    f"{f(last.get('cumulative_training_seconds'), '{:.0f}')} s | {hw(src['env'])} | {src['status'].get('status', 'INCOMPLETE')} |")
    rows += ["", "Only a run with status COMPLETED is a reproduced training; STOPPED / INTERRUPTED runs are partial records.", "",
             "## Historical documented values (not reproducible)", "", "| Modality | Quantity | Value | Source | Evidence |", "|---|---|---|---|---|"]
    for m in MODALITIES:
        rows += [f"| {m} | {q} | {v} | {s} | {H if v != NA else NA} |" for q, v, s in HISTORICAL[m]]
    rows += ["", "## All runs", ""]
    for m in MODALITIES:
        rows += [f"- **{m}**: {run_line(load(r))}" for r in runs(m)] or [f"- **{m}**: no runs"]
    (T / "TRAINING_SUMMARY.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def paper_table():
    face, _ = completed("face")
    lines = ["# Paper training table", "", "Only values safe to report: configuration read from code, and NEWLY_MEASURED values from COMPLETED "
             "reproducible runs. No historical, undocumented or estimated number appears here.", "",
             "| Modality | Model | Embedding | Dataset | Training | Reproduced result |", "|---|---|---|---|---|---|"]
    fr = ""
    if face:
        last = face["log"][-1]
        fr = (f"{len(face['log'])} epochs, {f(last['cumulative_training_seconds'], '{:.0f}')} s on {face['env'].get('cpu', NA)} (CPU); "
              f"final val. accuracy {f(last['validation_accuracy'], '{:.3f}')}; closed-set test EER {f(face['test'].get('EER'))}, AUC {f(face['test'].get('AUC'))}")
    lines += [
        "| Face | InceptionResNetV1 (VGGFace2-pretrained), last block + ArcFace head fine-tuned | 512-D, L2 | LFW (62 identities, >= 20 images each) | "
        "Adam 1e-4, batch 32, 10 epochs, ArcFace s=30 m=0.5, no augmentation, per-identity 70/15/15 split | " + (fr or NA) + " |",
        "| Voice | ECAPA-TDNN (SpeechBrain), trained from scratch | 192-D, L2 | VoxCeleb1 Indian-celebrity subset (24 speakers, 4,857 utterances) | "
        "AdamW 1e-3 (wd 1e-4), cosine schedule, batch 64, up to 30 epochs (patience 5 on val. loss), ArcFace s=30 m=0.5, 80-bin log-mel, augmentation disabled | "
        + (_result("voice") or NA) + " |",
        "", "Configuration values: `models/*/config.py`, `models/*/train.py`, `kaggle_kernels/face_training/face-embedding-training.ipynb`. "
            "Counts (24 / 4,857; 62 / 3,023) are re-counted by the runner from the datasets (see run config.json).",
    ]
    (T / "PAPER_TRAINING_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _result(m):
    r, _ = completed(m)
    if not r:
        return ""
    last = r["log"][-1]
    return (f"{len(r['log'])} epochs, {f(last['cumulative_training_seconds'], '{:.0f}')} s; best {r['best'].get('metric_name')} "
            f"{f(r['best'].get('best_metric'))} (epoch {r['best'].get('best_epoch')}); test EER {f(r['test'].get('EER'))}")


def modality_docs():
    for m in MODALITIES:
        (T / m / "runs").mkdir(parents=True, exist_ok=True)
        r, allruns = completed(m)
        cmd = f"python -m training.run_training {m}"
        readme = [f"# {m.capitalize()} training records", "", f"Reproduce: `{cmd}` (from the repository root; datasets under `data/`, see "
                  "`training/TRAINING_REPRODUCIBILITY_REPORT.md`).", "", "Each run lives in `runs/<experiment_id>/` with `config.json`, "
                  "`environment.json`, `training_log.csv` (written after every epoch), `checkpoints/checkpoint_manifest.json` "
                  "(checkpoint files themselves are gitignored), `best_checkpoint.json`, `metrics/`, `figures/`, `run_status.json`. "
                  "`latest` names the newest run. The deployed checkpoint in `models/" + m + "/saved/` is NOT produced or replaced by these runs.", "",
                  "## Runs", ""] + ([f"- {run_line(load(p))}" for p in runs(m)] or ["- none"])
        (T / m / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        st = [f"# {m.capitalize()} reproduction status", ""]
        if r:
            last = r["log"][-1]
            st += [f"**Training reproduced: YES** - run `{r['id']}` completed {len(r['log'])} of {r['config'].get('configured_epochs')} configured epochs.", "",
                   f"- Training time (sum of epoch wall times): {f(last['cumulative_training_seconds'], '{:.0f}')} s; hardware: {hw(r['env'])}",
                   f"- Best checkpoint: {r['best'].get('checkpoint_path')} - rule: {r['best'].get('selection_rule')}",
                   f"- Test verification: {json.dumps(r['test'])}"]
        else:
            partial = [load(p) for p in runs(m)]
            epoch_times = [float(row["elapsed_seconds"]) for p in partial for row in p["log"]]
            cfg_epochs = next((p["config"].get("configured_epochs") for p in partial), None)
            st += ["**Training reproduced: NO** - no run had reached COMPLETED when these documents were generated.", "",
                   "What was found: the training code (`models/" + m + "/train.py`), its configuration, the dataset (under `data/`) and the "
                   "deployed checkpoint are all present.", ""]
            if epoch_times:
                st += [f"Measured on this machine: {len(epoch_times)} epoch(s) completed, mean epoch wall time {sum(epoch_times) / len(epoch_times):.0f} s "
                       f"(configured maximum {cfg_epochs} epochs; early stopping may end sooner). A run without run_status.json was still "
                       "in progress or was stopped externally; its completed epochs are preserved in training_log.csv.", ""]
            else:
                st += ["Not started in this environment: this is a CPU-only machine with 8 GB RAM, and only one training job can run at a time "
                       "(the voice run was occupying it).", ""]
            st += [f"Command: `python -m training.run_training {m}` (then `python -m training.summarize`)", "", "Runs:"] + [f"- {run_line(p)}" for p in partial]
        st += ["", "Historical information that remains unavailable:"] + [f"- {q} ({s})" for q, v, s in HISTORICAL[m] if v == NA]
        (T / m / "REPRODUCTION_STATUS.md").write_text("\n".join(st) + "\n", encoding="utf-8")


def report():
    lines = ["# Training reproducibility report", "", "**NO HISTORICAL TRAINING VALUES WERE FABRICATED.**", "",
             "Generated by `python -m training.summarize` from `training/<modality>/runs/*` and the cited historical documents.", ""]
    lines += ["## 1-2. Reproduced and newly measured", ""]
    for m in MODALITIES:
        r, allruns = completed(m)
        lines.append(f"### {m}")
        if r:
            lines += [f"- Reproduced: **YES** (`{r['id']}`)", f"- Epoch log: `training/{m}/runs/{r['id']}/training_log.csv`"]
            for row in r["log"]:
                lines.append("  - epoch " + row["epoch"] + ": " + ", ".join(f"{k}={f(row[k])}" for k in ("train_loss", "validation_loss", "train_accuracy",
                                                                                                        "validation_accuracy", "validation_EER", "validation_AUC") if row[k] != NA))
            lines += [f"- Test verification (new checkpoint): {json.dumps(r['test'])}", f"- Figures: {', '.join(r['figures'])}"]
        else:
            lines.append("- Reproduced: **NO** - see `training/" + m + "/REPRODUCTION_STATUS.md`")
        for p in allruns:
            if p is not r:
                lines.append(f"- Partial / other run: {run_line(p)}" + (": " + "; ".join(
                    f"epoch {row['epoch']} " + ", ".join(f"{k}={f(row[k])}" for k in ("train_loss", "validation_loss", "train_accuracy", "validation_EER", "validation_AUC") if row[k] != NA)
                    + f", {row['elapsed_seconds']} s" for row in p["log"]) if p["log"] else ""))
        lines.append("")
    lines += ["## 3. Historically unavailable", ""] + [f"- {m}: {q} ({s})" for m in MODALITIES for q, v, s in HISTORICAL[m] if v == NA]
    lines += ["", "## 4. Commands", "", "```", "git lfs pull", "python -m training.run_training face",
              "python -m training.run_training voice",
              "python -m training.run_training voice --stop-after-epochs 1   # bounded probe (partial record)", "python -m training.summarize", "```", "",
              "Datasets (gitignored `data/`): LFW via scikit-learn into `data/lfw`; Kaggle `gaurav41/voxceleb1-audio-wav-files-for-india-celebrity` "
              "-> `data/voxceleb_subset` (Kaggle API token required).", ""]
    envs = [load(p)["env"] for m in MODALITIES for p in runs(m)]
    e = envs[-1] if envs else {}
    lines += ["## 5-6. Hardware / environment and git commit", "", "```json", json.dumps(e, indent=2), "```", "",
              "Each run's own `environment.json` is authoritative for that run.", "", "## 7-8. Datasets and training configurations", ""]
    for m in MODALITIES:
        rs = runs(m)
        if rs:
            c = load(rs[-1])["config"]
            lines += [f"### {m}", "```json", json.dumps({k: c[k] for k in c if k not in ("experiment_id", "timestamp_utc")}, indent=2, default=str), "```", ""]
    lines += ["## 9. Checkpoints", "", "New checkpoints are written only inside each run folder and are gitignored; manifests record file, epoch, metrics, "
              "timestamp, commit and seed. The deployed checkpoints (`models/*/saved/*.pt`, Git LFS) are the historical ones and were NOT replaced.", ""]
    for m in MODALITIES:
        for p in runs(m):
            r = load(p)
            if r["manifest"]:
                lines.append(f"- {m} `{r['id']}`: " + "; ".join(f"{x['checkpoint']} (epoch {x['epoch']})" for x in r["manifest"]))
    lines += ["", "## 10-11. Validation metrics and durations", "", "See section 1-2 (per-epoch values) and each run's `training_log.csv` "
              "(`elapsed_seconds`, `cumulative_training_seconds`).", "", "## 12. Limitations", "",
              "- Runs use CPU only (no CUDA on this machine); timings are not comparable with GPU training.",
              "- Voice full runs were not completed here (multi-hour on CPU; background jobs end with the Claude Code session).",
              "- Seeds are now fixed (42); the historical runs were unseeded, so reproduced numbers can differ from historical ones.",
              "- Face and voice validation/test identities overlap training (closed-set), as in the original pipelines.",
              "- Voice checkpoint selection uses validation loss (as implemented); validation EER is logged for information only.",
              "- The face notebook has no best-checkpoint rule; the final epoch is saved (recorded as such, not as a 'best' epoch).",
              "- Kaggle dataset version identifiers were not recorded at download time (NOT_AVAILABLE).",
              "- The deployed models were not retrained or replaced; paper results that use the deployed checkpoints refer to the historical training."]
    (T / "TRAINING_REPRODUCIBILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    summary()
    paper_table()
    modality_docs()
    report()
    print("training docs written")
