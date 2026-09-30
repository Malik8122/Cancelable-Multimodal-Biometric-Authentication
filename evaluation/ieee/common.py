"""Shared paths, evidence labels, metrics, statistics, table writers and the IEEE figure style."""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"  # gitignored: raw datasets and the embedding cache live only here
CACHE = DATA / "eval_cache"
EVAL = REPO / "evaluation"
RESULTS = EVAL / "results"
FIGURES = EVAL / "figures"
TABLES = EVAL / "tables"
REPORTS = EVAL / "reports"
for _d in (CACHE, RESULTS, FIGURES, TABLES, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# Exactly one of these per metric / artefact.
REAL = "REAL DATA"
SYNTHETIC = "SYNTHETIC CALIBRATION"
SOFTWARE = "SOFTWARE VALIDATION"
CONFIG = "CONFIGURATION"
DERIVED = "DERIVED"
FUTURE = "FUTURE WORK"
LABELS = (REAL, SYNTHETIC, SOFTWARE, CONFIG, DERIVED, FUTURE)

#: Evaluation-only HKDF root secret. Never the deployment MASTER_SECRET; keys derived from it protect nothing real.
EVAL_SECRET = "ieee-evaluation-secret-not-a-deployment-key"
SEED = 20260925


# ----------------------------------------------------------------------------- output


def write_csv(path: Path, rows: list[dict], label: str) -> Path:
    """Write rows with an `evidence_label` column (exactly one label per row)."""
    assert label in LABELS, label
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"note": "no rows"}]
    fields = ["evidence_label"] + [k for k in rows[0].keys() if k != "evidence_label"]
    for r in rows[1:]:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({"evidence_label": r.get("evidence_label", label), **{k: _fmt(v) for k, v in r.items() if k != "evidence_label"}})
    return path


def _fmt(value):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def write_table(name: str, rows: list[dict], label: str | None = None, title: str = "", note: str = "") -> None:
    """Paper table as CSV + Markdown. Rows must carry `evidence_label` and `source` unless `label` is given."""
    for r in rows:
        if label is not None:
            r.setdefault("evidence_label", label)
        assert r.get("evidence_label") in LABELS, (name, r)
        assert r.get("source"), (name, r)
    write_csv(TABLES / f"{name}.csv", rows, rows[0]["evidence_label"])
    cols = list(rows[0].keys())
    for r in rows[1:]:
        cols += [k for k in r if k not in cols]
    lines = [f"# {title or name}", ""]
    if note:
        lines += [note, ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(_fmt(r.get(c, ""))).replace("|", "\\|") for c in cols) + " |")
    (TABLES / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


@contextmanager
def timed(step: str):
    """Append the wall-clock runtime of a step to evaluation/reports/runtime_log.csv."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    log = REPORTS / "runtime_log.csv"
    new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        if new:
            w.writerow(["timestamp", "step", "seconds"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), step, f"{elapsed:.1f}"])
    print(f"[{step}] {elapsed:.1f}s", flush=True)


def hardware() -> dict:
    import psutil

    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "processor": platform.processor(),
    }
    try:
        import subprocess

        name = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        if name:
            info["processor"] = name
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["gpu"] = "none (CPU only)" if not torch.cuda.is_available() else torch.cuda.get_device_name(0)
        info["torch_threads"] = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        pass
    return info


# ----------------------------------------------------------------------------- verification metrics
# Convention: `higher_is_better=True` means accept iff score >= t; for distances pass higher_is_better=False
# (accept iff score <= t). Genuine = mated comparison, impostor = non-mated comparison.


def _orient(g, i, higher_is_better):
    g, i = np.asarray(g, float), np.asarray(i, float)
    return (g, i) if higher_is_better else (-g, -i)


def roc(genuine, impostor, higher_is_better: bool = True) -> dict:
    """Exact empirical ROC over every distinct score (vectorized). Thresholds are in the original score units."""
    g, i = _orient(genuine, impostor, higher_is_better)
    scores = np.concatenate([g, i])
    is_gen = np.concatenate([np.ones(len(g), bool), np.zeros(len(i), bool)])
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], is_gen[order]
    distinct = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = np.cumsum(~y)[distinct]
    tpr = np.r_[0.0, tp / len(g)]
    fpr = np.r_[0.0, fp / len(i)]
    thr = np.r_[np.inf, s[distinct]]
    if not higher_is_better:
        thr = -thr
    return {"fpr": fpr, "tpr": tpr, "threshold": thr}


def auc(curve: dict) -> float:
    return float(np.trapezoid(curve["tpr"], curve["fpr"]))


def eer(genuine, impostor, higher_is_better: bool = True) -> tuple[float, float]:
    """(EER, threshold): the ROC operating point where FAR and FRR are closest; EER = their mean there."""
    c = roc(genuine, impostor, higher_is_better)
    far, frr = c["fpr"], 1 - c["tpr"]
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2), float(c["threshold"][k])


def rates(genuine, impostor, threshold: float, higher_is_better: bool = True) -> dict:
    g, i = np.asarray(genuine, float), np.asarray(impostor, float)
    acc_g = g >= threshold if higher_is_better else g <= threshold
    acc_i = i >= threshold if higher_is_better else i <= threshold
    tp, fn = int(acc_g.sum()), int((~acc_g).sum())
    fp, tn = int(acc_i.sum()), int((~acc_i).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "threshold": threshold, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "FAR": fp / (fp + tn) if fp + tn else float("nan"),
        "FRR": fn / (fn + tp) if fn + tp else float("nan"),
        "TAR": recall,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": precision, "recall": recall,
        "F1": 2 * precision * recall / (precision + recall) if precision + recall else float("nan"),
    }


def tar_at_far(genuine, impostor, target_far: float, higher_is_better: bool = True) -> float:
    c = roc(genuine, impostor, higher_is_better)
    ok = c["fpr"] <= target_far
    return float(c["tpr"][ok].max()) if ok.any() else 0.0


def full_report(genuine, impostor, higher_is_better: bool, operating_threshold: float | None) -> dict:
    """EER, AUC, TAR@FAR, and every confusion-matrix metric at the EER threshold and at the operating threshold."""
    e, t_eer = eer(genuine, impostor, higher_is_better)
    c = roc(genuine, impostor, higher_is_better)
    out = {
        "n_genuine": len(genuine), "n_impostor": len(impostor), "EER": e, "EER_threshold": t_eer, "ROC_AUC": auc(c),
        "TAR_at_FAR_1e-2": tar_at_far(genuine, impostor, 1e-2, higher_is_better),
        "TAR_at_FAR_1e-3": tar_at_far(genuine, impostor, 1e-3, higher_is_better),
    }
    for k, v in rates(genuine, impostor, t_eer, higher_is_better).items():
        out[f"at_EER_{k}"] = v
    if operating_threshold is not None:
        for k, v in rates(genuine, impostor, operating_threshold, higher_is_better).items():
            out[f"at_operating_{k}"] = v
    return out


def describe(values) -> dict:
    """n, mean, SD, median, min, max, quartiles and a 95% CI of the mean (t distribution)."""
    from scipy import stats

    v = np.asarray(values, float)
    n = len(v)
    if n == 0:
        return {"n": 0}
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": n, "mean": float(v.mean()), "sd": sd, "median": float(med), "q1": float(q1), "q3": float(q3),
            "min": float(v.min()), "max": float(v.max()), "ci95_low": float(v.mean() - half), "ci95_high": float(v.mean() + half)}


def bootstrap_ci(genuine, impostor, fn, n_boot: int = 200, seed: int = SEED) -> tuple[float, float]:
    """95% percentile bootstrap CI of a (genuine, impostor) -> float metric, resampling both sets."""
    rng = np.random.default_rng(seed)
    g, i = np.asarray(genuine), np.asarray(impostor)
    vals = [fn(g[rng.integers(0, len(g), len(g))], i[rng.integers(0, len(i), len(i))]) for _ in range(n_boot)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ----------------------------------------------------------------------------- figures


def ieee_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "grid.linewidth": 0.4, "grid.alpha": 0.35,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


COL_W = 3.5  # IEEE single-column width, inches
DBL_W = 7.16  # IEEE double-column width
COLORS = {"face": "#1f4e79", "voice": "#b85c00", "raw": "#555555", "protected": "#c62828",
          "fusion": "#6a1b9a"}


def save_fig(fig, name: str, label: str, caption: str) -> Path:
    """Save a 300-dpi PNG and record its label + caption in figures/figure_index.csv."""
    assert label in LABELS
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    index = FIGURES / "figure_index.csv"
    rows = []
    if index.exists():
        with index.open(encoding="utf-8") as h:
            rows = [r for r in csv.DictReader(h) if r["file"] != path.name]
    rows.append({"file": path.name, "evidence_label": label, "caption": caption})
    with index.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["file", "evidence_label", "caption"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["file"]))
    return path
