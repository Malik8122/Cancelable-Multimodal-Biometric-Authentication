"""Fig. 3: ROC of the protected-template (256-bit BioHash) decision scores - face, voice and face+voice fusion - on
ONE protocol: the chimeric virtual users of evaluation/ieee/experiments.py part14
(20 pairings x 24 users x 3 attempts; LFW held-out faces x VoxCeleb1 test speakers).

Linear FPR/TPR axes with the random-classifier diagonal; an inset magnifies FPR 0-0.1. No title (IEEE: the caption
carries it). Legend EER / ROC-AUC are read from evaluation/results/fusion_score_level_eer.csv; the curves are regenerated from the
same deterministic scores and asserted to reproduce those values before anything is drawn.

Why not voice_roc.csv / face_metrics.csv: those are the training notebooks' RAW-embedding cosine
evaluations on different protocols (all-pairs; face_metrics.csv is a closed-set summary with no curve), so they are not
protected-template ROCs and are not comparable with the fusion curves.

Run: python -m evaluation.scripts.fig03_roc_protected_templates
Output: evaluation/figures/fig03_roc_protected_templates.png (300 dpi) and .pdf
"""

from __future__ import annotations

import csv

import numpy as np

from evaluation.ieee.common import COL_W, FIGURES, RESULTS, eer, ieee_style, roc
from evaluation.ieee.experiments import MODALITIES, chimeric_rows

#: CSV row name -> (legend label, colour, line style, line width, z-order); face+voice fusion is highlighted.
SERIES = {
    "face": ("Face", "#1f5fa8", "-", 1.1, 3),
    "voice": ("Voice", "#138a86", "-", 1.1, 3),
    "mean(face,voice)": ("Face + Voice fusion", "#c62828", "-", 2.0, 5),
}


def _reported() -> dict[str, tuple[float, float]]:
    with (RESULTS / "fusion_score_level_eer.csv").open(encoding="utf-8") as f:
        return {r["score"]: (float(r["EER"]), float(r["AUC"])) for r in csv.DictReader(f)}


def curves() -> dict[str, dict]:
    rows = chimeric_rows()
    gen = np.array([r["genuine"] for r in rows])
    scores = {m: np.array([r[f"{m}_score"] for r in rows]) for m in MODALITIES}
    scores["mean(face,voice)"] = (scores["face"] + scores["voice"]) / 2
    out = {}
    for name, s in scores.items():
        c = roc(s[gen], s[~gen])
        out[name] = {"curve": c, "EER": eer(s[gen], s[~gen])[0], "AUC": float(np.trapezoid(c["tpr"], c["fpr"])),
                     "n_genuine": int(gen.sum()), "n_impostor": int((~gen).sum())}
    return out


def main() -> None:
    reported = _reported()
    data = curves()
    for name in SERIES:  # the drawn curves must be the ones the CSV reports
        e, a = reported[name]
        assert abs(data[name]["EER"] - e) < 5e-6 and abs(data[name]["AUC"] - a) < 5e-6, (name, data[name]["EER"], e, data[name]["AUC"], a)

    plt = ieee_style()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.spines.top": True, "axes.spines.right": True})
    fig, ax = plt.subplots(figsize=(COL_W, 3.0), facecolor="white")
    ax.set_facecolor("white")
    ax.plot([0, 1], [0, 1], ls="--", color="#9e9e9e", lw=0.8, label="Random classifier", zorder=1)
    for name, (label, color, style, width, z) in SERIES.items():
        e, a = reported[name]  # legend values come from fusion_score_level_eer.csv
        c = data[name]["curve"]
        ax.plot(c["fpr"], c["tpr"], ls=style, color=color, lw=width, zorder=z, label=f"{label} (AUC {a:.4f}, EER {100 * e:.2f}%)")
    ax.set(xlim=(0, 1), ylim=(0, 1.005), xlabel="False Positive Rate (FPR)", ylabel="True Positive Rate (TPR)")
    ax.set_aspect("equal")
    ax.grid(True, color="#e0e0e0", lw=0.4)
    # zoom on the low-FPR region, where the curves separate (same data, same colours)
    ins = ax.inset_axes([0.36, 0.47, 0.40, 0.30])
    for name, (label, color, style, width, z) in SERIES.items():
        c = data[name]["curve"]
        ins.plot(c["fpr"], c["tpr"], ls=style, color=color, lw=width * 0.8, zorder=z)
    ins.set(xlim=(0, 0.1), ylim=(0.6, 1.0))
    ins.set_xticks([0, 0.025, 0.05, 0.075, 0.1], ["0", "0.025", "0.05", "0.075", "0.1"])
    ins.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ins.tick_params(labelsize=5, length=2, pad=1.5)
    ins.set_xlabel("FPR", fontsize=5, labelpad=1)
    ins.set_ylabel("TPR", fontsize=5, labelpad=1)
    ins.set_facecolor("white")
    ins.grid(True, color="#eeeeee", lw=0.3)
    ax.indicate_inset_zoom(ins, edgecolor="#808080", lw=0.5)
    ax.legend(loc="lower right", fontsize=5.6, frameon=True, framealpha=1.0, edgecolor="#cccccc", facecolor="white")
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig03_roc_protected_templates.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print("written:", *(FIGURES / f"fig03_roc_protected_templates.{e}" for e in ("png", "pdf")))


if __name__ == "__main__":
    main()
