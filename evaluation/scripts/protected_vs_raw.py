"""Raw embedding vs 256-bit protected template on IDENTICAL pairs (face, voice): the performance cost of protection.

RAW       = the exact embedding similarity (cosine), which the deployed system never computes at runtime.
PROTECTED = same-key Hamming similarity of the two 256-bit BioHash templates (what the deployed system compares).
Both are scored on the same cached enrollment-vs-probe comparisons (evaluation/ieee/experiments.comparisons).
EER and ROC-AUC are threshold-free, so Hamming similarity and its calibrated estimate give identical values.
Raw performance is NOT system performance.

Run: python -m evaluation.scripts.protected_vs_raw
"""

from __future__ import annotations

from evaluation.ieee.common import COL_W, COLORS, REAL, RESULTS, auc, bootstrap_ci, eer, ieee_style, roc, save_fig, write_csv
from evaluation.ieee.experiments import comparisons


def run():
    rows, curves = [], {}
    for modality in ("face", "voice"):
        c = comparisons(modality)
        for rep, v in (("raw_embedding_cosine", c.raw_cosine), ("protected_256bit_hamming", c.hamming)):
            g, i = v[c.genuine], v[~c.genuine]
            e, t = eer(g, i)
            lo, hi = bootstrap_ci(g, i, lambda a, b: eer(a, b)[0], n_boot=200)
            curve = roc(g, i)
            curves[(modality, rep)] = curve
            rows.append({"modality": modality, "representation": rep, "n_genuine": len(g), "n_impostor": len(i),
                         "EER": e, "EER_ci95_low": lo, "EER_ci95_high": hi, "EER_threshold": t, "ROC_AUC": auc(curve),
                         "identical_pairs": True})
        raw, prot = rows[-2], rows[-1]
        rows.append({"modality": modality, "representation": "delta_protected_minus_raw", "EER": prot["EER"] - raw["EER"],
                     "ROC_AUC": prot["ROC_AUC"] - raw["ROC_AUC"], "identical_pairs": True})
    write_csv(RESULTS / "protected_vs_raw.csv", rows, REAL)
    plt = ieee_style()
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    for modality in ("face", "voice"):
        for rep, style in (("raw_embedding_cosine", ":"), ("protected_256bit_hamming", "-")):
            cv = curves[(modality, rep)]
            ax.plot(cv["fpr"], cv["tpr"], style, color=COLORS[modality], label=f"{modality} {'raw' if rep.startswith('raw') else 'protected'}")
    ax.set(xscale="log", xlim=(1e-4, 1), ylim=(0, 1.01), xlabel="FAR", ylabel="TAR", title="Raw vs protected (identical pairs)")
    ax.grid(True, which="both")
    ax.legend(fontsize=6, loc="lower right")
    save_fig(fig, "protected_vs_raw_roc", REAL, "ROC of raw embedding cosine (dotted) vs 256-bit protected-template Hamming similarity (solid), identical pairs.")
    return rows


if __name__ == "__main__":
    for r in run():
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
