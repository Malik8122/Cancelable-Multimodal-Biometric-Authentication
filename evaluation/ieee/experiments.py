"""Experiments on real embeddings (Parts 3, 4, 6, 7, 8, 9, 14, 16 of the IEEE evaluation plan).

All scores come from the shipped transform (`protected.templates`, asserted bit-identical to production) and the
shipped decision rules (`protected.estimated_*`, asserted equal to backend/services/modality_metrics.decide).
Run: python -m evaluation.ieee.experiments [part ...]
"""

from __future__ import annotations

import json
import sys

import numpy as np

from evaluation.ieee.common import (
    CACHE, COL_W, COLORS, DBL_W, DERIVED, REAL, RESULTS, SEED, SYNTHETIC, bootstrap_ci, describe, eer, full_report,
    ieee_style, rates, roc, save_fig, save_json, timed, write_csv,
)
from evaluation.ieee.protected import (
    BITS, Comparisons, build_comparisons, estimated_cosine, estimated_distance, hamming_similarity, key_for, templates,
)

MODALITIES = ("face", "voice")

#: The deployed decision rule per modality: (score name, higher_is_better, threshold, Hamming-similarity equivalent).
SYSTEM_RULE = {
    "face": ("estimated_cosine", True, 0.80),
    "voice": ("estimated_distance", False, 0.75),
}
#: The teacher's metric computed EXACTLY on raw embeddings (what the estimate approximates).
RAW_RULE = {
    "face": ("raw_cosine", True, 0.80),
    "voice": ("raw_distance", False, 0.75),
}
CAL_JSON = RESULTS / "biohash_metric_calibration.json"


# ----------------------------------------------------------------------------- data


def load(modality: str) -> dict:
    z = np.load(CACHE / f"{modality}_embeddings.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def comparisons(modality: str) -> Comparisons:
    """Cached enrollment-vs-probe comparisons (see protected.build_comparisons)."""
    path = CACHE / f"{modality}_comparisons.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return Comparisons(modality=modality.split("_")[0], **{k: z[k] for k in z.files})
    d = load(modality.split("_")[0])
    E, labels = d["embeddings"], d["labels"]
    if modality == "face":
        c = build_comparisons("face", E, labels, impostors_per_identity=50)
    elif modality == "voice":
        c = build_comparisons("voice", E, labels, impostors_per_identity=None)
    else:
        raise KeyError(modality)
    np.savez_compressed(path, **{k: getattr(c, k) for k in ("target", "probe_identity", "genuine", "ref_index",
                                                             "probe_index", "raw_cosine", "hamming")})
    return c


def scores(c: Comparisons, modality: str) -> dict[str, np.ndarray]:
    return {
        "raw_cosine": c.raw_cosine,
        "raw_distance": np.sqrt(np.maximum(0, 2 - 2 * c.raw_cosine)),
        "hamming_similarity": c.hamming,
        "estimated_cosine": estimated_cosine(c.hamming, modality),
        "estimated_distance": estimated_distance(c.hamming, modality),
    }


# ----------------------------------------------------------------------------- Part 3: calibration on real embeddings


def _err_stats(true, est) -> dict:
    true, est = np.asarray(true), np.asarray(est)
    err = est - true
    ss_res, ss_tot = float((err**2).sum()), float(((true - true.mean()) ** 2).sum())
    return {"n": len(true), "RMSE": float(np.sqrt((err**2).mean())), "MAE": float(np.abs(err).mean()),
            "bias": float(err.mean()), "error_sd": float(err.std(ddof=1)), "pearson_r": float(np.corrcoef(true, est)[0, 1]),
            "R2": 1 - ss_res / ss_tot, "max_abs_error": float(np.abs(err).max())}


def part3_calibration_real():
    cal = json.loads(CAL_JSON.read_text())
    rows, binned = [], []
    plots = {}
    for modality in MODALITIES:
        c = comparisons(modality)
        s = scores(c, modality)
        true, est = c.raw_cosine, s["estimated_cosine"]
        clamp = est <= -0.2 + 1e-12
        curves = [(modality, est)]
        for curve_name, e in curves:
            for subset, mask in (("all", np.ones(len(true), bool)), ("genuine", c.genuine), ("impostor", ~c.genuine),
                                 ("in_calibrated_range", ~clamp)):
                row = {"modality": modality, "curve": curve_name, "subset": subset, **_err_stats(true[mask], e[mask]),
                       "fraction_clamped": float(clamp[mask].mean())}
                rows.append(row)
            d_true, d_est = s["raw_distance"], np.sqrt(np.maximum(0, 2 - 2 * e))
            rows.append({"modality": modality, "curve": curve_name, "subset": "euclidean_distance_all",
                         **_err_stats(d_true, d_est), "fraction_clamped": float(clamp.mean())})
        synth = cal["modalities"][modality]
        edges = np.arange(-0.2, 1.0001, 0.1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (true >= lo) & (true < hi) & ~clamp
            if m.sum() < 20:
                continue
            err = est[m] - true[m]
            centre = (lo + hi) / 2
            binned.append({"modality": modality, "true_cosine_bin": f"[{lo:.1f},{hi:.1f})", "n": int(m.sum()),
                           "mean_error": float(err.mean()), "error_sd": float(err.std(ddof=1)),
                           "synthetic_predicted_sd": float(np.interp(centre, synth["cosine"], synth["cosine_estimate_std"]))})
        plots[modality] = (true, est, c.genuine, clamp)
    write_csv(RESULTS / "calibration_real_validation.csv", rows, REAL)
    write_csv(RESULTS / "calibration_real_binned.csv", binned, REAL)
    _plot_calibration_real(plots, binned)


def _plot_calibration_real(plots, binned):
    plt = ieee_style()
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, len(MODALITIES), figsize=(DBL_W, 2.3))
    for ax, (m, (t, e, g, clamp)) in zip(axes, plots.items()):
        idx = rng.choice(len(t), min(6000, len(t)), replace=False)
        ax.scatter(t[idx][~g[idx]], e[idx][~g[idx]], s=1.5, alpha=0.25, color=COLORS["raw"], label="impostor pairs", rasterized=True)
        ax.scatter(t[idx][g[idx]], e[idx][g[idx]], s=1.5, alpha=0.4, color=COLORS[m], label="genuine pairs", rasterized=True)
        ax.plot([-0.2, 1], [-0.2, 1], "k--", lw=0.7, label="identity")
        ax.set(xlim=(-0.25, 1.02), ylim=(-0.25, 1.02), xlabel="true embedding cosine", ylabel="estimated cosine (from Hamming)", title=m)
        ax.grid(True)
    axes[0].legend(loc="upper left", markerscale=4)
    save_fig(fig, "fig21_calibration_real_scatter", REAL,
             "Estimated cosine (from the 256-bit template Hamming similarity) vs true embedding cosine on real pairs; dashed = identity.")

    fig, axes = plt.subplots(1, len(MODALITIES), figsize=(DBL_W, 2.1), sharey=True)
    for ax, m in zip(axes, plots):
        b = [r for r in binned if r["modality"] == m]
        x = [float(r["true_cosine_bin"][1:].split(",")[0]) + 0.05 for r in b]
        ax.errorbar(x, [r["mean_error"] for r in b], yerr=[r["error_sd"] for r in b], fmt="o", ms=2.5, color=COLORS[m],
                    capsize=2, label="real: mean error ± SD")
        ax.plot(x, [r["synthetic_predicted_sd"] for r in b], "k:", lw=0.8, label="synthetic ±SD (calibration)")
        ax.plot(x, [-r["synthetic_predicted_sd"] for r in b], "k:", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set(xlabel="true embedding cosine", title=m)
        ax.grid(True)
    axes[0].set_ylabel("estimate − true")
    axes[0].legend(loc="lower right")
    save_fig(fig, "fig22_calibration_real_error_vs_cosine", REAL,
             "Estimation error by true-cosine bin on real pairs (mean ± SD) against the SD predicted by the synthetic calibration.")

    fig, axes = plt.subplots(1, len(MODALITIES), figsize=(DBL_W, 1.9), sharey=False)
    for ax, (m, (t, e, g, clamp)) in zip(axes, plots.items()):
        err = (e - t)[~clamp]
        ax.hist(err, bins=80, color=COLORS[m], alpha=0.85)
        ax.axvline(0, color="k", lw=0.6)
        ax.set(xlabel="estimate − true cosine", ylabel="comparisons", title=m)
    save_fig(fig, "fig23_calibration_real_residual_hist", REAL, "Histogram of estimation residuals on real pairs (clamped estimates excluded).")


# ----------------------------------------------------------------------------- Part 4: raw vs protected


def part4_raw_vs_protected():
    rows, curves = [], {}
    for modality in MODALITIES:
        c = comparisons(modality)
        s = scores(c, modality)
        for representation, (name, hib, thr) in (("raw_embedding", RAW_RULE[modality]), ("protected_template", SYSTEM_RULE[modality])):
            g, i = s[name][c.genuine], s[name][~c.genuine]
            rep = full_report(g, i, hib, thr)
            lo, hi = bootstrap_ci(g, i, lambda a, b: eer(a, b, hib)[0], n_boot=100)
            rows.append({"modality": modality, "representation": representation, "score": name,
                         "operating_threshold": "" if thr is None else thr, "operating_rule": "" if thr is None else (">=" if hib else "<="),
                         **rep, "EER_ci95_low": lo, "EER_ci95_high": hi})
            curves[(modality, representation)] = roc(g, i, hib)
    write_csv(RESULTS / "raw_vs_protected_metrics.csv", rows, REAL)
    _plot_roc_det(curves)
    return rows


def _plot_roc_det(curves):
    from scipy.stats import norm

    plt = ieee_style()
    for modality in MODALITIES:
        fig, ax = plt.subplots(figsize=(COL_W * 0.8, 2.3))
        for rep, style in (("raw_embedding", "-"), ("protected_template", "--")):
            c = curves[(modality, rep)]
            ax.plot(c["fpr"], c["tpr"], style, color=COLORS["raw" if rep.startswith("raw") else "protected"],
                    label=f"{'raw embedding' if rep.startswith('raw') else '256-bit protected template'}")
        ax.set(xscale="log", xlim=(1e-4, 1), ylim=(0, 1.01), xlabel="FAR", ylabel="TAR", title=f"{modality}: ROC")
        ax.grid(True, which="both")
        ax.legend(loc="lower right")
        save_fig(fig, f"fig{ {'face': 12, 'voice': 13}[modality] }_roc_{modality}_raw_vs_protected", REAL,
                 f"ROC of the {modality} pipeline on real data: raw embedding cosine vs 256-bit protected-template Hamming similarity.")
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    for modality in MODALITIES:
        c = curves[(modality, "protected_template")]
        ax.plot(c["fpr"], c["tpr"], color=COLORS[modality], label=f"{modality} (protected)")
        c = curves[(modality, "raw_embedding")]
        ax.plot(c["fpr"], c["tpr"], ":", color=COLORS[modality], label=f"{modality} (raw)")
    ax.set(xscale="log", xlim=(1e-4, 1), ylim=(0, 1.01), xlabel="FAR", ylabel="TAR", title="ROC: protected templates vs raw")
    ax.grid(True, which="both")
    ax.legend(fontsize=6, loc="lower right")
    save_fig(fig, "fig14_roc_protected_templates", REAL, "ROC of all modalities, protected templates (solid) vs raw embeddings (dotted), real data.")
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ticks = [0.001, 0.01, 0.05, 0.2, 0.5]
    for modality in MODALITIES:
        for rep, style in (("protected_template", "-"), ("raw_embedding", ":")):
            c = curves[(modality, rep)]
            far, frr = np.clip(c["fpr"], 1e-5, 1 - 1e-5), np.clip(1 - c["tpr"], 1e-5, 1 - 1e-5)
            ax.plot(norm.ppf(far), norm.ppf(frr), style, color=COLORS[modality], label=f"{modality} ({rep.split('_')[0]})")
    ax.set_xticks(norm.ppf(ticks), [f"{t:g}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks), [f"{t:g}" for t in ticks])
    ax.set(xlim=norm.ppf([5e-4, 0.6]), ylim=norm.ppf([5e-4, 0.6]), xlabel="FAR", ylabel="FRR", title="DET curves")
    ax.grid(True)
    ax.legend(fontsize=6)
    save_fig(fig, "fig15_det_curves", REAL, "DET curves (normal-deviate axes), protected (solid) vs raw (dotted), real data.")


# ----------------------------------------------------------------------------- Part 6: threshold sweep


SWEEP = {
    "face": ("estimated_cosine", "raw_cosine", True, np.round(np.arange(0.20, 0.9501, 0.01), 3)),  # requested 0.70-0.95 + down to the EER point
    "voice": ("estimated_distance", "raw_distance", False, np.round(np.arange(0.50, 1.2001, 0.02), 3)),
}


def part6_threshold_sweep():
    rows, eers = [], []
    for modality, (prot, raw, hib, grid) in SWEEP.items():
        c = comparisons(modality)
        s = scores(c, modality)
        for rep, name in (("protected_template", prot), ("raw_embedding", raw)):
            if name is None:
                continue
            g, i = s[name][c.genuine], s[name][~c.genuine]
            e, t = eer(g, i, hib)
            eers.append({"modality": modality, "representation": rep, "score": name, "EER": e, "EER_threshold": t})
            for thr in grid:
                r = rates(g, i, float(thr), hib)
                rows.append({"modality": modality, "representation": rep, "score": name, "rule": ">=" if hib else "<=",
                             **{k: r[k] for k in ("threshold", "FAR", "FRR", "TAR", "accuracy", "precision", "F1")}})
    write_csv(RESULTS / "threshold_sweep.csv", rows, REAL)
    write_csv(RESULTS / "threshold_sweep_eer.csv", eers, REAL)
    _plot_sweep(rows, eers)


def _plot_sweep(rows, eers):
    plt = ieee_style()
    configured = {m: SYSTEM_RULE[m][2] for m in MODALITIES}
    fig, axes = plt.subplots(1, len(MODALITIES), figsize=(DBL_W, 2.2))
    for ax, m in zip(axes, MODALITIES):
        for rep, style in (("protected_template", "-"), ("raw_embedding", ":")):
            rr = [r for r in rows if r["modality"] == m and r["representation"] == rep]
            if not rr:
                continue
            x = [r["threshold"] for r in rr]
            ax.plot(x, [r["FAR"] for r in rr], style, color="#c62828", label=f"FAR ({rep.split('_')[0]})")
            ax.plot(x, [r["FRR"] for r in rr], style, color="#1f4e79", label=f"FRR ({rep.split('_')[0]})")
        e = next(r for r in eers if r["modality"] == m and r["representation"] == "protected_template")
        ax.axvline(configured[m], color="k", lw=0.7, ls="--", label="configured threshold")
        ax.plot([e["EER_threshold"]], [e["EER"]], "ko", ms=3, label=f"EER {e['EER']:.3f}")
        ax.set(xlabel={"face": "est. cosine threshold (accept ≥)", "voice": "est. distance threshold (accept ≤)"}[m],
               ylabel="rate", title=m, ylim=(0, 1))
        ax.grid(True)
    axes[0].legend(fontsize=5.5)
    save_fig(fig, "fig16_threshold_sensitivity", REAL, "FAR and FRR vs decision threshold (protected: solid; exact raw metric: dotted); dot = protected EER.")
    fig, axes = plt.subplots(1, len(MODALITIES), figsize=(DBL_W, 1.9))
    for ax, m in zip(axes, MODALITIES):
        rr = [r for r in rows if r["modality"] == m and r["representation"] == "protected_template"]
        x = [r["threshold"] for r in rr]
        ax.plot(x, [r["accuracy"] for r in rr], color=COLORS[m], label="accuracy")
        ax.plot(x, [r["F1"] for r in rr], "--", color=COLORS[m], label="F1")
        ax.axvline(configured[m], color="k", lw=0.7, ls="--")
        ax.set(xlabel="threshold", title=m, ylim=(0, 1.02))
        ax.grid(True)
    axes[0].legend()
    save_fig(fig, "fig25_accuracy_f1_vs_threshold", REAL,
             "Accuracy and F1 of the protected-template decision vs threshold (depend on the genuine:impostor ratio of the protocol).")


# ----------------------------------------------------------------------------- Parts 7 & 9: template sets and revocation (real code path)


def _harness(user_id: str, pool: int = 4):
    from evaluation.template_set_experiments import _Harness

    return _Harness(pool_size=pool, user_id=user_id)


def part7_9_template_sets_and_revocation(max_users: int = 150):
    """Real embeddings through the real ModalityService + crud template-set lifecycle on an in-memory SQLite DB."""
    from backend.config import Settings
    from backend.services.modality_metrics import decide
    from template_protection.utils import unpack_bits

    s0 = Settings(master_secret="x")
    # the template-space equivalent of each deployed rule (face est. cosine >= 0.80, voice est. distance <= 0.75)
    thresholds = {m: decide(m, 0.9, BITS, s0).hamming_threshold for m in ("face", "voice")}
    per_attempt, summary = [], []
    rng = np.random.default_rng(SEED)
    for modality in ("face", "voice"):
        d = load(modality)
        E, labels = d["embeddings"], d["labels"]
        idents = [u for u in sorted(set(labels.tolist())) if (labels == u).sum() >= 3]
        rng.shuffle(idents)
        for ident in idents[:max_users]:
            idx = np.nonzero(labels == ident)[0]
            ref, probe = E[idx[0]], E[idx[1]]
            h = _harness(f"{modality}-{ident}")
            h.enroll({modality: ref})
            sets ={r.template_set_version: unpack_bits(r.protected_template, num_bits=r.output_bits)
                    for r in h.rows() if r.modality == modality}
            before = h.authenticate({modality: probe})[modality]
            old_active = sets[1]
            for a in range(1, 5):
                for b in range(a + 1, 5):
                    per_attempt.append({"modality": modality, "user": ident, "measure": "cross_set_same_embedding",
                                        "similarity": float(hamming_similarity(sets[a], sets[b]))})
            per_attempt.append({"modality": modality, "user": ident, "measure": "within_key_genuine_before_revocation",
                                "similarity": before.hamming_similarity, "accepted": before.authenticated})
            promoted_ok = []
            for step in range(3):  # walk sets 1 -> 2 -> 3 -> 4
                h.revoke()
                after = h.authenticate({modality: probe})[modality]
                promoted_ok.append(after.authenticated)
                per_attempt.append({"modality": modality, "user": ident, "measure": f"genuine_after_revocation_{step + 1}",
                                    "similarity": after.hamming_similarity, "accepted": after.authenticated})
            new_active = sets[4]
            replay = float(hamming_similarity(old_active, new_active))
            per_attempt.append({"modality": modality, "user": ident, "measure": "revoked_template_vs_new_active",
                                "similarity": replay, "accepted": replay >= thresholds[modality]})
            try:
                h.revoke()
                exhausted = False
            except Exception:  # noqa: BLE001 - TemplatePoolExhaustedError expected
                exhausted = True
            from evaluation.template_set_experiments import _APPLICATION_ID
            h.services[modality].enroll(h.db, ref, h.user_id, _APPLICATION_ID)  # re-enrollment after exhaustion
            re = h.authenticate({modality: probe})[modality]
            per_attempt.append({"modality": modality, "user": ident, "measure": "genuine_after_reenrollment",
                                "similarity": re.hamming_similarity, "accepted": re.authenticated, "pool_exhausted_before": exhausted})
    write_csv(RESULTS / "revocation_per_attempt.csv", per_attempt, REAL)
    for modality in ("face", "voice"):
        for measure in sorted({r["measure"] for r in per_attempt}):
            rr = [r for r in per_attempt if r["modality"] == modality and r["measure"] == measure]
            acc = [r["accepted"] for r in rr if "accepted" in r]
            summary.append({"modality": modality, "measure": measure, **describe([r["similarity"] for r in rr]),
                            "acceptance_rate": float(np.mean(acc)) if acc else "", "hamming_threshold": thresholds[modality]})
    write_csv(RESULTS / "revocation_summary.csv", summary, REAL)
    _plot_revocation(per_attempt, thresholds)


def _plot_revocation(rows, thresholds):
    plt = ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W * 0.75, 2.0), sharey=True)
    bins = np.linspace(0.3, 1.0, 57)
    for ax, m in zip(axes, ("face", "voice")):
        for measure, color, lab in (("within_key_genuine_before_revocation", COLORS[m], "genuine, before revocation"),
                                    ("genuine_after_revocation_1", "#2e7d32", "genuine, after revocation"),
                                    ("revoked_template_vs_new_active", COLORS["protected"], "revoked vs new template")):
            v = [r["similarity"] for r in rows if r["modality"] == m and r["measure"] == measure]
            ax.hist(v, bins=bins, alpha=0.6, color=color, label=lab)
        ax.axvline(thresholds[m], color="k", ls="--", lw=0.7, label="decision threshold")
        ax.set(xlabel="Hamming similarity", title=m)
    axes[0].set_ylabel("users")
    axes[0].legend(fontsize=6)
    save_fig(fig, "fig18_template_revocation_histogram", REAL,
             "Revocation on real embeddings through the shipped template-set code: genuine similarity before/after revocation and the revoked template vs the new active template.")


# ----------------------------------------------------------------------------- Part 8: unlinkability (Gomez-Barrero et al., TIFS 2018)


def dsys(mated, non_mated, bin_width: float, omega: float = 1.0):
    """D<->(s) and D_sys from histogram densities (no KDE). Returns (centres, D_local, D_sys, p_m, p_nm)."""
    edges = np.arange(0, 1 + bin_width, bin_width)
    pm, _ = np.histogram(mated, bins=edges, density=True)
    pn, _ = np.histogram(non_mated, bins=edges, density=True)
    d_local = np.zeros_like(pm)
    only_mated = (pm > 0) & (pn == 0)  # LR = infinity: fully linkable at this score
    d_local[only_mated] = 1.0
    both = (pm > 0) & (pn > 0)
    lr = pm[both] / pn[both]
    d_local[both] = np.where(omega * lr > 1, 2 * omega * lr / (1 + omega * lr) - 1, 0.0)  # empty bins stay 0
    d_sys = float(np.sum(d_local * pm * bin_width))
    return (edges[:-1] + edges[1:]) / 2, d_local, d_sys, pm, pn


def part8_unlinkability():
    rows, dists = [], {}
    for modality in MODALITIES:
        c = comparisons(modality)
        d = load(modality)
        E, labels = d["embeddings"], d["labels"]
        # template of each sample under its OWN identity's key in two different applications (databases A and B)
        TA, TB = {}, {}
        for ident in sorted(set(labels.tolist())):
            idx = np.nonzero(labels == ident)[0]
            TA.update(zip(idx.tolist(), templates(E[idx], key_for(ident, modality, application_id="db-A"))))
            TB.update(zip(idx.tolist(), templates(E[idx], key_for(ident, modality, application_id="db-B"))))
        mated = np.array([hamming_similarity(TA[r], TB[p]) for r, p in zip(c.ref_index[c.genuine], c.probe_index[c.genuine])])
        non = np.array([hamming_similarity(TA[r], TB[p]) for r, p in zip(c.ref_index[~c.genuine], c.probe_index[~c.genuine])])
        same_sample = np.array([hamming_similarity(TA[r], TB[r]) for r in np.unique(c.ref_index)])
        rng = np.random.default_rng(SEED)

        def null_floor(n, bw, reps=20):
            """D_sys between two random draws of the NON-mated set (same sizes): the estimator's noise floor."""
            vals = []
            for _ in range(reps):
                perm = rng.permutation(len(non))
                vals.append(dsys(non[perm[:n]], non[perm[n:]], bw)[2])
            return float(np.mean(vals)), float(np.percentile(vals, 95))

        for bw in (2 / 256, 4 / 256, 8 / 256):
            _, _, ds, _, _ = dsys(mated, non, bw)
            floor_mean, floor_p95 = null_floor(len(mated), bw)
            rows.append({"modality": modality, "bin_width": bw, "D_sys": ds, "n_mated": len(mated), "n_non_mated": len(non),
                         "null_floor_mean": floor_mean, "null_floor_p95": floor_p95, "exceeds_null_p95": ds > floor_p95})
        _, _, ds_same, _, _ = dsys(same_sample, non, 4 / 256)
        floor_mean, floor_p95 = null_floor(len(same_sample), 4 / 256)
        rows.append({"modality": modality, "bin_width": 4 / 256, "D_sys": ds_same, "n_mated": len(same_sample),
                     "n_non_mated": len(non), "null_floor_mean": floor_mean, "null_floor_p95": floor_p95,
                     "exceeds_null_p95": ds_same > floor_p95, "note": "mated = the SAME sample under two keys (worst case)"})
        for name, v in (("mated_cross_key", mated), ("non_mated_cross_key", non), ("same_sample_cross_key", same_sample)):
            rows.append({"modality": modality, "distribution": name, **describe(v)})
        dists[modality] = (mated, non)
    write_csv(RESULTS / "unlinkability.csv", rows, REAL)
    plt = ieee_style()
    fig, axes = plt.subplots(1, len(MODALITIES), figsize=(DBL_W, 2.1))
    fig.subplots_adjust(wspace=0.75)
    for k, (ax, (m, (mated, non))) in enumerate(zip(axes, dists.items())):
        centres, dl, ds, pm, pn = dsys(mated, non, 4 / 256)
        floor = next(r["null_floor_mean"] for r in rows if r["modality"] == m and r.get("bin_width") == 4 / 256 and not r.get("note"))
        ax.bar(centres, pn, width=4 / 256, alpha=0.5, color=COLORS["raw"], label="non-mated")
        ax.bar(centres, pm, width=4 / 256, alpha=0.5, color=COLORS[m], label="mated")
        ax.set(xlim=(0.38, 0.64), xlabel="cross-key Hamming similarity", title=f"{m}: $D_{{sys}}$={ds:.3f} (null {floor:.3f})")
        if k == 0:
            ax.set_ylabel("density")
        ax2 = ax.twinx()
        ax2.plot(centres, dl, "k-", lw=0.7)
        ax2.set_ylim(0, 1.05)
        ax2.spines["right"].set_visible(True)
        if k == len(MODALITIES) - 1:
            ax2.set_ylabel(r"local $D_{\leftrightarrow}(s)$")
    axes[0].legend(fontsize=6, loc="upper left")
    save_fig(fig, "fig19_unlinkability_histogram", REAL,
             "Unlinkability (Gomez-Barrero et al. 2018): mated vs non-mated cross-key scores on real embeddings, local D<-> (line), D_sys and its permutation null floor (histogram estimate, bin 4/256).")


# ----------------------------------------------------------------------------- Part 14: fusion (chimeric virtual users)


def chimeric_rows(pairings: int = 20, attempts: int = 3) -> list[dict]:
    """Per-attempt protected-template decisions and fusion scores for chimeric users (independent face and voice
    identities combined into virtual subjects; standard multibiometric practice when no true multimodal
    corpus exists; assumes modality independence). Deterministic (seed SEED)."""
    from backend.config import Settings
    from backend.services.modality_metrics import decide

    settings = Settings(master_secret="x")
    data = {m: load(m) for m in MODALITIES}
    voice_ids = sorted(set(data["voice"]["labels"].tolist()))
    n_users = len(voice_ids)

    def pools(modality):
        d = data[modality]
        refs, probes = {}, {}
        for lab in sorted(set(d["labels"].tolist())):
            idx = np.nonzero(d["labels"] == lab)[0]
            refs[lab], probes[lab] = idx[0], list(idx[1:])
        return {lab: (refs[lab], probes[lab]) for lab in refs if len(probes[lab]) >= attempts}

    pool = {m: pools(m) for m in MODALITIES}
    rng = np.random.default_rng(SEED)
    all_rows = []
    for p in range(pairings):
        chosen = {"face": rng.choice(sorted(pool["face"]), n_users, replace=False)}
        chosen["voice"] = np.array(voice_ids)
        users = [{m: chosen[m][u] for m in MODALITIES} for u in range(n_users)]
        # probe attempts: attempt k of user u uses the k-th probe of each of u's identities
        probe_of = {(u, k): {m: pool[m][users[u][m]][1][k] for m in MODALITIES} for u in range(n_users) for k in range(attempts)}
        attempts_list = list(probe_of.items())
        for t, target in enumerate(users):
            h = {}
            for m in MODALITIES:  # one keyed projection per (target, modality): reference + every probe in one batch
                idx = [pool[m][target[m]][0]] + [probe[m] for _, probe in attempts_list]
                T = templates(data[m]["embeddings"][idx], key_for(f"{p}-{t}", m))
                h[m] = hamming_similarity(T[1:], T[0][None, :])
            for j, ((u, k), _) in enumerate(attempts_list):
                row = {"pairing": p, "target": t, "claimant": u, "attempt": k, "genuine": u == t}
                for m in MODALITIES:
                    dec = decide(m, float(h[m][j]), BITS, settings)
                    row[f"{m}_value"], row[f"{m}_match"], row[f"{m}_score"], row[f"{m}_thr"] = dec.value, dec.matched, dec.fusion_score, dec.fusion_threshold
                all_rows.append(row)
    return all_rows


def part14_fusion(pairings: int = 20, attempts: int = 3):
    """Fusion policies and score-level fusion on chimeric users (see chimeric_rows)."""
    from fusion.config import FusionPolicy
    from fusion.policy import evaluate_fusion_policy

    all_rows = chimeric_rows(pairings, attempts)
    n_users = len(set(load("voice")["labels"].tolist()))
    policies = {
        "face_only": (("face",), lambda r: r["face_match"]),
        "voice_only": (("voice",), lambda r: r["voice_match"]),
    }

    def with_policy(mods, policy):
        def f(r):
            return evaluate_fusion_policy({m: r[f"{m}_score"] for m in mods}, {m: r[f"{m}_match"] for m in mods}, policy,
                                          fusion_threshold=float(np.mean([r[f"{m}_thr"] for m in mods]))).authenticated
        return f

    fv = ("face", "voice")
    policies.update({
        "ALL_REQUIRED(face+voice)": (fv, with_policy(fv, FusionPolicy.ALL_REQUIRED)),
        "WEIGHTED(face+voice)": (fv, with_policy(fv, FusionPolicy.WEIGHTED)),
        "OR(face+voice)": (fv, lambda r: r["face_match"] or r["voice_match"]),
        "MAJORITY(face+voice)=AND": (fv, lambda r: r["face_match"] and r["voice_match"]),
    })
    out = []
    for name, (mods, fn) in policies.items():
        per = []
        for p in range(pairings):
            rr = [r for r in all_rows if r["pairing"] == p]
            acc = np.array([fn(r) for r in rr])
            gen = np.array([r["genuine"] for r in rr])
            tp, fn_, fp, tn = int((acc & gen).sum()), int((~acc & gen).sum()), int((acc & ~gen).sum()), int((~acc & ~gen).sum())
            per.append({"FAR": fp / (fp + tn), "FRR": fn_ / (fn_ + tp), "TAR": tp / (tp + fn_), "accuracy": (tp + tn) / len(rr),
                        "precision": tp / (tp + fp) if tp + fp else np.nan, "TP": tp, "FN": fn_, "FP": fp, "TN": tn})
        agg = {"policy": name, "modalities": "+".join(mods), "pairings": pairings, "users_per_pairing": n_users,
               "genuine_per_pairing": n_users * attempts, "impostor_per_pairing": n_users * (n_users - 1) * attempts}
        for k in ("FAR", "FRR", "TAR", "accuracy", "precision"):
            v = np.array([x[k] for x in per], float)
            agg[f"{k}_mean"], agg[f"{k}_sd"] = float(np.nanmean(v)), float(np.nanstd(v, ddof=1))
        for k in ("TP", "FN", "FP", "TN"):
            agg[f"{k}_total"] = int(sum(x[k] for x in per))
        tp, fp, fn_ = agg["TP_total"], agg["FP_total"], agg["FN_total"]
        agg["F1_pooled"] = 2 * tp / (2 * tp + fp + fn_)
        out.append(agg)
    write_csv(RESULTS / "fusion_policy_metrics.csv", out, REAL)
    # score-level ROC: single modalities vs mean fusion score
    gen = np.array([r["genuine"] for r in all_rows])
    curves = {}
    for m in MODALITIES:
        s = np.array([r[f"{m}_score"] for r in all_rows])
        curves[m] = (roc(s[gen], s[~gen]), eer(s[gen], s[~gen])[0])
    for name, mods in (("mean(face,voice)", fv),):
        s = np.mean([[r[f"{m}_score"] for r in all_rows] for m in mods], axis=0)
        curves[name] = (roc(s[gen], s[~gen]), eer(s[gen], s[~gen])[0])
    write_csv(RESULTS / "fusion_score_level_eer.csv", [{"score": k, "EER": v[1], "AUC": float(np.trapezoid(v[0]["tpr"], v[0]["fpr"]))}
                                                      for k, v in curves.items()], REAL)
    plt = ieee_style()
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    for k, (c, e) in curves.items():
        style = {"mean(face,voice)": ("#6a1b9a", "--")}.get(k, (COLORS.get(k), "-"))
        ax.plot(c["fpr"], c["tpr"], color=style[0], ls=style[1], label=f"{k} (EER {e:.3f})")
    ax.set(xscale="log", xlim=(1e-3, 1), ylim=(0, 1.01), xlabel="FAR", ylabel="TAR", title="Score-level fusion (chimeric users)")
    ax.grid(True, which="both")
    ax.legend(fontsize=6, loc="lower right")
    save_fig(fig, "fig26_fusion_roc", REAL, "ROC of single protected modalities vs equal-weight mean fusion score, chimeric virtual users from real data.")
    return out


# ----------------------------------------------------------------------------- Part 16: template security statistics


def part16_security():
    from scipy.stats import binom

    rows = []
    from backend.config import Settings
    from backend.services.modality_metrics import decide

    s0 = Settings(master_secret="x")
    for modality in MODALITIES:
        d = load(modality)
        E, labels = d["embeddings"], d["labels"]
        ref_idx = [int(np.nonzero(labels == u)[0][0]) for u in sorted(set(labels.tolist()))]
        T = np.vstack([templates(E[i], key_for(labels[i], modality))[0] for i in ref_idx])  # each user under own key
        n = len(T)
        ones = T.mean(axis=0)
        H = -(ones * np.log2(np.clip(ones, 1e-12, 1)) + (1 - ones) * np.log2(np.clip(1 - ones, 1e-12, 1)))
        R = np.corrcoef(T.T.astype(float))
        off = np.abs(R[~np.eye(BITS, dtype=bool)])
        rng = np.random.default_rng(SEED)
        a, b = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
        keep = a != b
        hd = (T[a[keep]] != T[b[keep]]).mean(axis=1)  # cross-user, cross-key normalized Hamming distance
        p, sd = float(hd.mean()), float(hd.std(ddof=1))
        dof = p * (1 - p) / sd**2
        thr_h = decide(modality, 0.9, BITS, s0).hamming_threshold
        max_bits = int(np.floor(BITS * (1 - thr_h) + 1e-9))
        empirical_collision = float((hd * BITS <= max_bits).mean())
        binom_collision = float(binom.cdf(np.floor(max_bits / BITS * dof), int(round(dof)), p))
        # key rotation: same reference embedding, key_version 1 vs 2
        rot = np.array([hamming_similarity(templates(E[i], key_for(labels[i], modality, 1))[0], templates(E[i], key_for(labels[i], modality, 2))[0]) for i in ref_idx])
        rows += [
            {"modality": modality, "metric": "templates_analyzed", "value": n, "evidence_label": REAL},
            {"modality": modality, "metric": "mean_fraction_of_ones", "value": float(ones.mean()), "evidence_label": REAL},
            {"modality": modality, "metric": "min_position_fraction_of_ones", "value": float(ones.min()), "evidence_label": REAL},
            {"modality": modality, "metric": "max_position_fraction_of_ones", "value": float(ones.max()), "evidence_label": REAL},
            {"modality": modality, "metric": "mean_hamming_weight_bits", "value": float(T.sum(axis=1).mean()), "evidence_label": REAL},
            {"modality": modality, "metric": "mean_per_bit_entropy_bits", "value": float(H.mean()), "evidence_label": REAL},
            {"modality": modality, "metric": "sum_per_bit_entropy_upper_bound_bits", "value": float(H.sum()), "evidence_label": DERIVED,
             "note": "upper bound; assumes independent bits"},
            {"modality": modality, "metric": "mean_abs_pairwise_bit_correlation", "value": float(off.mean()), "evidence_label": REAL},
            {"modality": modality, "metric": "expected_mean_abs_correlation_if_independent", "value": float(np.sqrt(2 / (np.pi * n))), "evidence_label": DERIVED},
            {"modality": modality, "metric": "fraction_bit_pairs_abs_corr_gt_0.1", "value": float((off > 0.1).mean()), "evidence_label": REAL},
            {"modality": modality, "metric": "cross_user_cross_key_mean_normalized_HD", "value": p, "evidence_label": REAL},
            {"modality": modality, "metric": "cross_user_cross_key_sd_normalized_HD", "value": sd, "evidence_label": REAL},
            {"modality": modality, "metric": "degrees_of_freedom_daugman", "value": dof, "evidence_label": DERIVED,
             "note": "N = p(1-p)/sigma^2 from the cross-user cross-key HD distribution"},
            {"modality": modality, "metric": "decision_max_differing_bits", "value": max_bits, "evidence_label": DERIVED},
            {"modality": modality, "metric": "empirical_cross_key_collision_rate_at_threshold", "value": empirical_collision,
             "evidence_label": REAL, "note": f"fraction of {int(keep.sum())} cross-user cross-key pairs within {max_bits} bits"},
            {"modality": modality, "metric": "binomial_cross_key_collision_estimate_at_threshold", "value": binom_collision,
             "evidence_label": DERIVED, "note": "binomial model with the DoF above"},
            {"modality": modality, "metric": "key_rotation_same_embedding_mean_similarity", "value": float(rot.mean()), "evidence_label": REAL},
            {"modality": modality, "metric": "key_rotation_same_embedding_sd_similarity", "value": float(rot.std(ddof=1)), "evidence_label": REAL},
        ]
    path = RESULTS / "template_security.csv"
    write_csv(path, rows, REAL)


PARTS = {
    "3": part3_calibration_real, "4": part4_raw_vs_protected, "6": part6_threshold_sweep,
    "7_9": part7_9_template_sets_and_revocation, "8": part8_unlinkability, "14": part14_fusion, "16": part16_security,
}

if __name__ == "__main__":
    for part in sys.argv[1:] or list(PARTS):
        with timed(f"part{part}"):
            PARTS[part]()
