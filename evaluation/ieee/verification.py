"""Part 2: verify every committed metric.

(a) recompute each metric in evaluation/results/*_metrics.csv from that modality's own confusion matrix / ROC CSV;
(b) reproduce the committed protocol from freshly extracted embeddings (same split, same all-pairs protocol);
(c) re-run the committed template-set experiments (same seeds) and compare, WITHOUT overwriting the committed CSVs;
(d) recompute the calibration fit statistics from the calibration JSON.
Run: python -m evaluation.ieee.verification
Outputs: evaluation/results/metric_verification.csv, evaluation/metric_verification_report.md
"""

from __future__ import annotations

import csv
import json

import numpy as np

from evaluation.ieee.common import CACHE, DERIVED, EVAL, REAL, RESULTS, SYNTHETIC, auc, eer, rates, roc, timed, write_csv
from template_protection.metric_estimation import fitted_hamming_similarity


def _kv(path):
    with open(path, encoding="utf-8") as h:
        return {r["metric"]: r["value"] for r in csv.DictReader(h)}


def _fv(path):
    with open(path, encoding="utf-8") as h:
        return {r["field"]: r["value"] for r in csv.DictReader(h)}


def _cm(path):
    with open(path, encoding="utf-8") as h:
        rows = list(csv.reader(h))
    tn, fp = int(rows[1][1]), int(rows[1][2])
    fn, tp = int(rows[2][1]), int(rows[2][2])
    return tp, fn, fp, tn


def _roc_auc(path):
    with open(path, encoding="utf-8") as h:
        r = list(csv.DictReader(h))
    return float(np.trapezoid([float(x["tpr"]) for x in r], [float(x["fpr"]) for x in r])), len(r)


def check(rows, source, metric, reported, recomputed, method, label, tol=1e-4):
    rep = float(reported) if reported not in (None, "") else None
    rec = float(recomputed) if recomputed is not None else None
    status = "NOT VERIFIABLE" if rep is None or rec is None else ("MATCH" if abs(rep - rec) <= tol else "DISCREPANCY")
    rows.append({"source": source, "metric": metric, "reported": reported, "recomputed": "" if rec is None else rec,
                 "abs_difference": "" if status == "NOT VERIFIABLE" else abs(rep - rec), "status": status, "method": method,
                 "evidence_label": label})


def run():
    rows = []
    # (a) internal consistency of the committed CSVs
    for m in ("voice",):
        rep = _kv(RESULTS / f"{m}_metrics.csv")
        tp, fn, fp, tn = _cm(RESULTS / f"{m}_confusion_matrix.csv")
        a, npts = _roc_auc(RESULTS / f"{m}_roc.csv")
        src = f"evaluation/results/{m}_metrics.csv"
        recomputed = {"accuracy": (tp + tn) / (tp + tn + fp + fn), "far": fp / (fp + tn), "frr": fn / (fn + tp),
                      "precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn),
                      "num_genuine_pairs": tp + fn, "num_impostor_pairs": fp + tn, "auc": a}
        n = int(float(rep["num_samples"]))
        recomputed["num_pairs_total_vs_C(n,2)"] = n * (n - 1) // 2
        rep["num_pairs_total_vs_C(n,2)"] = int(float(rep["num_genuine_pairs"])) + int(float(rep["num_impostor_pairs"]))
        for k, v in recomputed.items():
            check(rows, src, k, rep.get(k), v, f"from {m}_confusion_matrix.csv" if k != "auc" else f"trapezoid over {npts} points of {m}_roc.csv", DERIVED)
        check(rows, src, "TAR", rep.get("recall"), tp / (tp + fn), "TAR = recall = TP/(TP+FN)", DERIVED)
        check(rows, src, "eer_vs_(far+frr)/2", rep.get("eer"), (fp / (fp + tn) + fn / (fn + tp)) / 2, "EER point of the committed confusion matrix", DERIVED)
    face = _kv(RESULTS / "face_metrics.csv")
    for k in ("accuracy", "eer", "auc"):
        check(rows, "evaluation/results/face_metrics.csv", k, face.get(k), None, "no pair-level / ROC data committed for face", DERIVED)

    # (b) reproduction from freshly extracted embeddings, committed protocol = all pairs of the test set
    for m, label_key in (("voice", "labels"),):
        z = np.load(CACHE / f"{m}_embeddings.npz", allow_pickle=True)
        E, L = z["embeddings"], z[label_key]
        S = E @ E.T
        iu = np.triu_indices(len(E), 1)
        same = (L[:, None] == L[None, :])[iu]
        s = S[iu]
        g, i = s[same], s[~same]
        e, t = eer(g, i)
        rep = _kv(RESULTS / f"{m}_metrics.csv")
        src = f"evaluation/results/{m}_metrics.csv (reproduction)"
        r = rates(g, i, t)
        check(rows, src, "num_samples", rep["num_samples"], len(E), "re-extracted test set", REAL, tol=0)
        check(rows, src, "num_genuine_pairs", rep["num_genuine_pairs"], len(g), "all pairs, same labels", REAL, tol=0)
        check(rows, src, "eer", rep["eer"], e, "all-pairs raw cosine, re-extracted embeddings", REAL, tol=5e-3)
        check(rows, src, "auc", rep["auc"], auc(roc(g, i)), "all-pairs raw cosine, re-extracted embeddings", REAL, tol=5e-3)
        check(rows, src, "eer_threshold", rep["eer_threshold"], t, "all-pairs raw cosine", REAL, tol=5e-3)
        check(rows, src, "accuracy_at_eer", rep["accuracy"], r["accuracy"], "at the reproduced EER threshold", REAL, tol=5e-3)

    # (c) template-set experiments re-run with the committed seeds
    from evaluation import template_set_experiments as tse

    for name, fn, csvname in (("diversity", tse.experiment_template_set_diversity, "template_set_diversity.csv"),
                              ("revocation", tse.experiment_template_set_revocation, "template_set_revocation.csv"),
                              ("promotion", tse.experiment_template_set_promotion, "template_set_promotion.csv"),
                              ("exhaustion", tse.experiment_template_set_exhaustion, "template_set_exhaustion.csv")):
        rerun = fn()
        committed = _fv(RESULTS / csvname)
        for k, v in rerun.items():
            if k in committed and isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
                check(rows, f"evaluation/results/{csvname} (re-run)", k, committed[k], v, "same function, same default seed", SYNTHETIC, tol=1e-9)

    # (d) calibration JSON statistics
    cal = json.loads((RESULTS / "biohash_metric_calibration.json").read_text())
    for m, e in cal["modalities"].items():
        c = np.array(e["cosine"])
        res = fitted_hamming_similarity(c, e["coefficients"]) - np.array(e["hamming_similarity_mean"])
        check(rows, "evaluation/results/biohash_metric_calibration.json", f"{m}_max_fit_residual", e["max_fit_residual"],
              float(np.abs(res).max()), "fitted model vs stored means", SYNTHETIC, tol=1e-5)
    write_csv(RESULTS / "metric_verification.csv", rows, DERIVED)
    _report(rows)
    return rows


def _report(rows):
    counts = {s: sum(r["status"] == s for r in rows) for s in ("MATCH", "DISCREPANCY", "NOT VERIFIABLE")}
    lines = ["# Metric verification report", "", "Generated by `python -m evaluation.ieee.verification`.", "",
             f"Checks: {len(rows)} - MATCH {counts['MATCH']}, DISCREPANCY {counts['DISCREPANCY']}, NOT VERIFIABLE {counts['NOT VERIFIABLE']}.", "",
             "Tolerances: 1e-4 for internal consistency, 5e-3 for reproduction from re-extracted embeddings, exact for re-runs.", "",
             "| source | metric | reported | recomputed | abs diff | status | method | label |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rec = "" if r["recomputed"] == "" else f"{r['recomputed']:.6g}"
        diff = "" if r["abs_difference"] == "" else f"{r['abs_difference']:.2g}"
        lines.append(f"| {r['source']} | {r['metric']} | {r['reported']} | {rec} | {diff} | {r['status']} | {r['method']} | {r['evidence_label']} |")
    (EVAL / "metric_verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    with timed("part2_verification"):
        run()
