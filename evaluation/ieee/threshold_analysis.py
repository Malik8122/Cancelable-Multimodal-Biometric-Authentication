"""Threshold sensitivity and operating-point analysis of the DEPLOYED protected-template decision (face, voice).

Pipeline scored: embedding -> L2 -> 256-bit BioHash (target identity's key) -> Hamming similarity -> calibrated estimate
(face: estimated cosine, accept >= t; voice: estimated Euclidean distance, accept <= t). Comparisons are the cached
evaluation/ieee enrollment-vs-probe pairs (held-out LFW identities; VoxCeleb1 test speakers).

Protocol for operating-point selection (no tuning on the test set):
  identities are split 50/50 (seeded) into DEVELOPMENT and TEST; a comparison belongs to a split only if BOTH its
  enrolled identity and its probe identity are in that split (cross-split impostor pairs are dropped).
  Candidate operating points are chosen on DEVELOPMENT only and then evaluated ONCE on TEST.
Nothing here changes the configured thresholds (THRESHOLD_SOURCE stays TEACHER_REQUESTED_BASELINE).

Run: python -m evaluation.ieee.threshold_analysis
"""

from __future__ import annotations

import numpy as np

from evaluation.ieee.common import COL_W, COLORS, DBL_W, REAL, RESULTS, SEED, eer, ieee_style, rates, roc, save_fig, write_csv
from evaluation.ieee.experiments import comparisons, scores

#: modality -> (protected score, raw score, higher_is_better, teacher threshold, sweep grid)
RULES = {
    "face": ("estimated_cosine", "raw_cosine", True, 0.80, np.round(np.arange(0.60, 0.9001, 0.01), 2)),
    "voice": ("estimated_distance", "raw_distance", False, 0.75, np.round(np.arange(0.50, 1.0001, 0.01), 2)),
}
FAR_TARGETS = (1e-2, 1e-3)


def split_masks(c, seed: int = SEED):
    """(dev, test) boolean masks over comparisons: identity-disjoint halves."""
    ids = np.array(sorted(set(c.target.tolist()) | set(c.probe_identity.tolist())), dtype=object)
    perm = np.random.default_rng(seed).permutation(len(ids))
    dev_ids = set(ids[perm[: len(ids) // 2]].tolist())
    t_dev = np.array([t in dev_ids for t in c.target])
    p_dev = np.array([p in dev_ids for p in c.probe_identity])
    return t_dev & p_dev, ~t_dev & ~p_dev


def threshold_for_far(genuine, impostor, target_far: float, higher_is_better: bool) -> float:
    """The most permissive threshold whose FAR on these scores is <= target_far."""
    c = roc(genuine, impostor, higher_is_better)
    ok = np.nonzero(c["fpr"] <= target_far)[0]
    return float(c["threshold"][ok[np.argmax(c["tpr"][ok])]])


def _metrics(g, i, thr, hib) -> dict:
    r = rates(g, i, float(thr), hib)
    return {k: r[k] for k in ("threshold", "TP", "FN", "FP", "TN", "FAR", "FRR", "TAR", "accuracy", "precision", "recall", "F1")}


def run():
    sweep_rows, op_rows, curves = [], [], {}
    for modality, (prot, raw, hib, teacher, grid) in RULES.items():
        c = comparisons(modality)
        s = scores(c, modality)
        dev, test = split_masks(c)
        subsets = {"all": np.ones(len(c.genuine), bool), "development": dev, "test": test}
        for rep, name in (("protected_template", prot), ("raw_embedding", raw)):
            for subset, m in subsets.items():
                g, i = s[name][m & c.genuine], s[name][m & ~c.genuine]
                e, et = eer(g, i, hib)
                curves[(modality, rep, subset)] = roc(g, i, hib)
                for thr in grid:
                    sweep_rows.append({"modality": modality, "representation": rep, "score": name, "rule": ">=" if hib else "<=",
                                       "subset": subset, "n_genuine": len(g), "n_impostor": len(i), **_metrics(g, i, thr, hib),
                                       "subset_EER": e, "subset_EER_threshold": et})
        # operating points: selected on DEVELOPMENT (protected scores), evaluated once on TEST
        gd, idv = s[prot][dev & c.genuine], s[prot][dev & ~c.genuine]
        gt, it = s[prot][test & c.genuine], s[prot][test & ~c.genuine]
        candidates = [("TEACHER_REQUESTED_BASELINE", teacher, "project requirement (not data-selected)")]
        candidates.append(("dev_EER_point", eer(gd, idv, hib)[1], "FAR = FRR on development"))
        for f in FAR_TARGETS:
            candidates.append((f"dev_FAR<={f:g}", threshold_for_far(gd, idv, f, hib), f"most permissive threshold with development FAR <= {f:g}"))
        for label, thr, how in candidates:
            for subset, (g, i) in (("development", (gd, idv)), ("test", (gt, it))):
                op_rows.append({"modality": modality, "score": prot, "rule": ">=" if hib else "<=", "operating_point": label,
                                "selection": how, "selected_on": "n/a" if label.startswith("TEACHER") else "development",
                                "evaluated_on": subset, "n_genuine": len(g), "n_impostor": len(i), **_metrics(g, i, thr, hib)})
    write_csv(RESULTS / "threshold_sensitivity.csv", sweep_rows, REAL)
    write_csv(RESULTS / "threshold_operating_points.csv", op_rows, REAL)
    _plots(sweep_rows, curves)
    return sweep_rows, op_rows


def _plots(rows, curves):
    from scipy.stats import norm

    plt = ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.3))
    for ax, (modality, (_, _, hib, teacher, _)) in zip(axes, RULES.items()):
        rr = [r for r in rows if r["modality"] == modality and r["representation"] == "protected_template" and r["subset"] == "all"]
        x = [r["threshold"] for r in rr]
        ax.plot(x, [r["FAR"] for r in rr], color="#c62828", label="FAR")
        ax.plot(x, [r["FRR"] for r in rr], color="#1f4e79", label="FRR")
        ax.plot(x, [r["TAR"] for r in rr], "--", color="#2e7d32", label="TAR")
        ax.axvline(teacher, color="k", lw=0.7, ls=":", label=f"teacher baseline {teacher}")
        ax.set(xlabel=f"{'estimated cosine (accept >=)' if hib else 'estimated distance (accept <=)'}", ylabel="rate", ylim=(0, 1.02),
               title=f"{modality}: protected-template decision")
        ax.grid(True)
    axes[0].legend(fontsize=6)
    save_fig(fig, "threshold_far_frr_tar", REAL, "FAR, FRR and TAR vs threshold of the deployed protected-template decision (face 0.60-0.90, voice 0.50-1.00).")
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.3))
    ticks = [0.001, 0.01, 0.05, 0.2, 0.5]
    for modality, color in (("face", COLORS["face"]), ("voice", COLORS["voice"])):
        for subset, style in (("development", "-"), ("test", "--")):
            c = curves[(modality, "protected_template", subset)]
            axes[0].plot(c["fpr"], c["tpr"], style, color=color, label=f"{modality} {subset}")
            far, frr = np.clip(c["fpr"], 1e-5, 1 - 1e-5), np.clip(1 - c["tpr"], 1e-5, 1 - 1e-5)
            axes[1].plot(norm.ppf(far), norm.ppf(frr), style, color=color, label=f"{modality} {subset}")
    axes[0].set(xscale="log", xlim=(1e-4, 1), ylim=(0, 1.01), xlabel="FAR", ylabel="TAR", title="ROC (protected)")
    axes[1].set_xticks(norm.ppf(ticks), [f"{t:g}" for t in ticks])
    axes[1].set_yticks(norm.ppf(ticks), [f"{t:g}" for t in ticks])
    axes[1].set(xlim=norm.ppf([5e-4, 0.6]), ylim=norm.ppf([5e-4, 0.6]), xlabel="FAR", ylabel="FRR", title="DET (protected)")
    for ax in axes:
        ax.grid(True, which="both")
        ax.legend(fontsize=6)
    save_fig(fig, "threshold_roc_det_dev_test", REAL, "ROC and DET of the protected-template decision on the identity-disjoint development and test splits.")


if __name__ == "__main__":
    _, ops = run()
    for r in ops:
        print(f"{r['modality']:5} {r['operating_point']:28} on {r['evaluated_on']:11} thr={r['threshold']:.4f} "
              f"FAR={r['FAR']:.5f} FRR={r['FRR']:.4f} TAR={r['TAR']:.4f}")
