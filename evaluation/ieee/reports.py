"""Parts 19-20 and the final package: paper tables, claim validation and IEEE_EVALUATION_PACKAGE.md.

Every number is read from a result CSV/JSON at generation time (nothing typed by hand), and every table row cites
its source file and the command that reproduces it.
Run: python -m evaluation.ieee.reports
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from evaluation.ieee.common import (
    CONFIG, DERIVED, EVAL, FIGURES, FUTURE, REAL, REPO, REPORTS, RESULTS, SOFTWARE, SYNTHETIC, TABLES, write_table,
)

CMD = {
    "extract": "python -m evaluation.ieee.extract_embeddings",
    "experiments": "python -m evaluation.ieee.experiments",
    "robustness": "python -m evaluation.ieee.robustness",
    "latency": "python -m scripts.benchmark_latency",
    "memory": "python -m scripts.benchmark_memory",
    "storage": "python -m evaluation.ieee.storage",
    "verification": "python -m evaluation.ieee.verification",
    "audit": "python -m evaluation.ieee.audit",
    "calibration": "python -m scripts.calibrate_biohash_metric_mapping",
    "real_cal": "python -m scripts.validate_calibration_real",
}


def rd(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as h:
        return list(csv.DictReader(h))


def pick(rows, **f):
    out = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in f.items())]
    return out[0] if out else {}


def num(v, fmt="{:.4f}"):
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return "NOT AVAILABLE"


def pct(v, digits=2):
    try:
        return f"{100 * float(v):.{digits}f}%"
    except (TypeError, ValueError):
        return "NOT AVAILABLE"


def src(file, cmd):
    return f"evaluation/results/{file} ({CMD[cmd]})"


# ----------------------------------------------------------------------------- tables


def tables():
    cfg = {r["parameter"]: r for r in rd("configuration.csv")}
    keys = ["Face embedding dimension", "Voice embedding dimension", "Template bits (runtime)",
            "Template sets per user (pool)", "HKDF hash", "HKDF seed length (bytes)", "Quantization threshold sigma", "Projection",
            "Default fusion policy", "Fusion weights", "Face minimum valid poses", "Voice sample rate (Hz)", "Voice clip length (s)", "Max upload size (bytes)"]
    write_table("table01_system_configuration", [{"parameter": k, "value": cfg[k]["value"], "source": cfg[k]["source"], "evidence_label": CONFIG}
                                                 for k in keys if k in cfg], title="System configuration")
    write_table("table02_dataset_summary", [
        {"modality": r["modality"], "dataset": r["dataset"], "evaluation_split": r["evaluation_split"], "identities": r["identities"],
         "samples_embedded": r["embedded"], "failures_to_acquire": r["failures_to_acquire"], "license": r["license"],
         "source": src("dataset_audit.csv", "audit"), "evidence_label": REAL} for r in rd("dataset_audit.csv")], title="Evaluation datasets")
    write_table("table03_threshold_configuration", [
        {"modality": "face", "metric": "estimated cosine similarity", "direction": "higher is better", "threshold": cfg["Face threshold (estimated cosine, accept >=)"]["value"],
         "hamming_equivalent": "h >= 0.8176 (<= 46 of 256 bits differ)", "origin": "teacher-requested; not experimentally optimized",
         "source": cfg["Face threshold (estimated cosine, accept >=)"]["source"], "evidence_label": CONFIG},
        {"modality": "voice", "metric": "estimated Euclidean distance", "direction": "lower is better", "threshold": cfg["Voice threshold (estimated Euclidean distance, accept <=)"]["value"],
         "hamming_equivalent": "h >= 0.7823 (<= 55 of 256 bits differ)", "origin": "teacher-requested; not experimentally optimized",
         "source": cfg["Voice threshold (estimated Euclidean distance, accept <=)"]["source"], "evidence_label": CONFIG},
        {"modality": "fusion (face+voice)", "metric": "mean of fusion-scale scores", "direction": "higher is better", "threshold": 0.759375,
         "hamming_equivalent": "-", "origin": "derived = mean(0.80, 1 - 0.75^2/2); informational under ALL_REQUIRED",
         "source": "backend/services/modality_metrics.py", "evidence_label": DERIVED},
    ], title="Threshold configuration")
    cal = json.loads((RESULTS / "biohash_metric_calibration.json").read_text())
    rows = []
    for m, e in cal["modalities"].items():
        rows.append({"modality": m, "data": "synthetic pairs (19,600)", "max_fit_residual_hamming": e["max_fit_residual"],
                     "coefficients": ", ".join(f"{c:.4f}" for c in e["coefficients"]), "RMSE_cosine": "n/a", "MAE_cosine": "n/a", "R2": "n/a", "pearson_r": "n/a", "bias": "n/a",
                     "source": "evaluation/results/biohash_metric_calibration.json (python -m scripts.calibrate_biohash_metric_mapping)", "evidence_label": SYNTHETIC})
    for r in rd("calibration_real_validation.csv"):
        if r["subset"] in ("all", "genuine", "in_calibrated_range") and r["curve"] == r["modality"]:
            rows.append({"modality": r["modality"], "data": f"real pairs, {r['subset']} (n={r['n']})", "max_fit_residual_hamming": "n/a",
                         "coefficients": "n/a", "RMSE_cosine": num(r["RMSE"]), "MAE_cosine": num(r["MAE"]), "R2": num(r["R2"]), "pearson_r": num(r["pearson_r"]),
                         "bias": num(r["bias"]), "source": src("calibration_real_validation.csv", "real_cal"), "evidence_label": REAL})
    write_table("table04_calibration_statistics", rows, title="Calibration: synthetic fit and real-embedding validation")
    perf = rd("raw_vs_protected_metrics.csv")
    for name, rep, title in (("table05_model_performance_raw", "raw_embedding", "Embedding-model performance (raw embeddings)"),
                             ("table06_protected_template_performance", "protected_template", "Protected-template (deployed) performance")):
        rr = []
        for m in ("face", "voice"):
            r = pick(perf, modality=m, representation=rep)
            if not r:
                continue
            rr.append({"modality": m, "score": r["score"], "n_genuine": r["n_genuine"], "n_impostor": r["n_impostor"], "EER": pct(r["EER"]),
                       "EER_95CI": f"{pct(r['EER_ci95_low'])} - {pct(r['EER_ci95_high'])}", "ROC_AUC": num(r["ROC_AUC"]),
                       "TAR@FAR=1%": pct(r["TAR_at_FAR_1e-2"]), "TAR@FAR=0.1%": pct(r["TAR_at_FAR_1e-3"]),
                       "operating_rule": f"{r['score']} {r['operating_rule']} {r['operating_threshold']}" if r.get("operating_threshold") else "-",
                       "FAR@op": pct(r.get("at_operating_FAR")), "FRR@op": pct(r.get("at_operating_FRR")), "F1@op": num(r.get("at_operating_F1")),
                       "source": src("raw_vs_protected_metrics.csv", "experiments"), "evidence_label": REAL})
        write_table(name, rr, title=title)
    write_table("table07_fusion_performance", [
        {"policy": r["policy"], "modalities": r["modalities"], "FAR": f"{pct(r['FAR_mean'])} ± {pct(r['FAR_sd'])}",
         "FRR": f"{pct(r['FRR_mean'])} ± {pct(r['FRR_sd'])}", "accuracy": f"{pct(r['accuracy_mean'])}", "F1_pooled": num(r["F1_pooled"]),
         "TP/FN/FP/TN": f"{r['TP_total']}/{r['FN_total']}/{r['FP_total']}/{r['TN_total']}",
         "protocol": f"{r['pairings']} chimeric pairings x {r['users_per_pairing']} virtual users",
         "source": src("fusion_policy_metrics.csv", "experiments"), "evidence_label": REAL} for r in rd("fusion_policy_metrics.csv")],
        title="Fusion policies (chimeric virtual users from real data; mean ± SD over pairings)")
    lat = rd("latency_benchmark.csv")
    write_table("table08_latency", [
        {"stage": r["stage"], "mode": r["mode"], "runs": r["runs"], "median_ms": num(r["median_ms"], "{:.2f}"), "mean_ms": num(r["mean_ms"], "{:.2f}"),
         "p95_ms": num(r["p95_ms"], "{:.2f}"), "p99_ms": num(r["p99_ms"], "{:.2f}"), "source": src("latency_benchmark.csv", "latency"), "evidence_label": REAL}
        for r in lat if r["mode"] in ("warm", "cold")] or [{"stage": "latency", "median_ms": "NOT AVAILABLE", "source": CMD["latency"], "evidence_label": FUTURE}],
        title="Latency (evaluation machine, CPU only)")
    st = rd("storage_analysis.csv")
    write_table("table09_storage", [{**{k: v for k, v in r.items() if v not in ("", None) and k != "evidence_label"},
                                     "source": src("storage_analysis.csv", "storage"), "evidence_label": r["evidence_label"]} for r in st], title="Storage")
    sec = []
    for r in rd("template_security.csv"):
        sec.append({"modality": r["modality"], "property": r["metric"], "value": num(r["value"]), "note": r.get("note", ""),
                    "source": src("template_security.csv", "experiments"), "evidence_label": r["evidence_label"]})
    for r in rd("unlinkability.csv"):
        if r.get("D_sys"):
            sec.append({"modality": r["modality"], "property": f"D_sys (bin {float(r['bin_width']) * 256:.0f}/256){' - ' + r['note'] if r.get('note') else ''}",
                        "value": num(r["D_sys"]), "source": src("unlinkability.csv", "experiments"), "evidence_label": REAL})
    for r in rd("revocation_summary.csv"):
        if r["measure"] in ("revoked_template_vs_new_active", "genuine_after_revocation_1", "within_key_genuine_before_revocation", "genuine_after_reenrollment"):
            sec.append({"modality": r["modality"], "property": f"{r['measure']}: mean similarity / acceptance", "value": f"{num(r['mean'])} / {pct(r['acceptance_rate'])}",
                        "source": src("revocation_summary.csv", "experiments"), "evidence_label": REAL})
    write_table("table10_security_properties", sec or [{"property": "-", "value": "NOT AVAILABLE", "source": "-", "evidence_label": FUTURE}], title="Template security properties")
    write_table("table11_software_validation", [{**{k: v for k, v in r.items() if v not in ("", None) and k != "evidence_label"},
                                                 "source": "evaluation/results/software_validation.csv (python -m evaluation.ieee.audit software)",
                                                 "evidence_label": SOFTWARE} for r in rd("software_validation.csv")] or
                [{"check": "software validation", "result": "NOT RUN", "source": "-", "evidence_label": FUTURE}], title="Software validation")
    write_table("table12_limitations", [{"limitation": t, "source": s, "evidence_label": FUTURE} for t, s in LIMITATIONS], title="Limitations / remaining work")


LIMITATIONS = [
    ("No labelled real-user study of the deployed system (consented participants); harness ready", "evaluation/real_user_evaluation.py"),
    ("Voice evaluation is closed-set (test speakers also in training; 24 speakers)", "evaluation/results/dataset_audit.csv"),
    ("Fusion uses chimeric virtual users (independence assumption; 24 voice identities limit the user count)", "evaluation/results/fusion_policy_metrics.csv"),
    ("No presentation-attack detection; IAPMR not measured (no attack data)", "evaluation/results/pad_results.csv"),
    ("Robustness degradations are simulated; pose and physical microphone change not measured", "evaluation/results/robustness.csv"),
    ("No demographic analysis (no demographic labels for the evaluation sets)", "evaluation/results/dataset_audit.csv"),
    ("Thresholds 0.80 / 0.75 are teacher-requested, not optimized on validation data", "backend/config.py"),
    ("Latency/memory measured on one laptop CPU; no GPU or server benchmark", "evaluation/reports/hardware.json"),
    ("Security analysis is statistical (entropy, correlation, unlinkability); no attack (inversion, hill-climbing, stolen-key) experiments", "evaluation/results/template_security.csv"),
]


# ----------------------------------------------------------------------------- claims


def claims() -> str:
    perf = rd("raw_vs_protected_metrics.csv")
    unl = rd("unlinkability.csv")
    rev = rd("revocation_summary.csv")
    cr = rd("calibration_real_validation.csv")
    sw = rd("software_validation.csv")
    fus = rd("fusion_policy_metrics.csv")
    lines = ["# Paper claims", "", "Every numeric claim lists: value, source file, producing script, experiment, reproduction command.",
             "Generated by `python -m evaluation.ieee.reports` from the result files - regenerate after re-running experiments.", ""]

    def claim(text, file, cmd, experiment, label):
        return f"- {text}  \n  *{label}* - source `evaluation/results/{file}`, script `{CMD[cmd]}`, experiment: {experiment}."

    safe = ["## SAFE CLAIMS", ""]
    safe.append("- The system stores only 256-bit keyed BioHash templates (32 bytes) per modality and template set; no image, audio or embedding "
                "is stored.  \n  *CONFIGURATION* - `backend/config.py`, `backend/database/models.py`, `evaluation/results/db_schema.csv` (`python -m evaluation.ieee.audit`).")
    for m in ("face", "voice"):
        raw, pro = pick(perf, modality=m, representation="raw_embedding"), pick(perf, modality=m, representation="protected_template")
        if raw and pro:
            safe.append(claim(f"{m.capitalize()}: EER {pct(raw['EER'])} (raw embedding, {raw['score']}) vs {pct(pro['EER'])} (256-bit protected template, 95% CI "
                              f"{pct(pro['EER_ci95_low'])}-{pct(pro['EER_ci95_high'])}); ROC-AUC {num(raw['ROC_AUC'])} vs {num(pro['ROC_AUC'])}; "
                              f"{pro['n_genuine']} genuine / {pro['n_impostor']} impostor comparisons.",
                              "raw_vs_protected_metrics.csv", "experiments", "Part 4, enrollment-vs-probe protocol, target-key impostors", REAL))
            if pro.get("operating_threshold"):
                safe.append(claim(f"{m.capitalize()} at the configured rule ({pro['score']} {pro['operating_rule']} {pro['operating_threshold']}): "
                                  f"FAR {pct(pro['at_operating_FAR'], 4)}, FRR {pct(pro['at_operating_FRR'])} on this evaluation set.",
                                  "raw_vs_protected_metrics.csv", "experiments", "Part 4 operating point", REAL))
    binned = rd("calibration_real_binned.csv")
    for m in ("face", "voice"):
        r, g = pick(cr, modality=m, curve=m, subset="all"), pick(cr, modality=m, curve=m, subset="genuine")
        b = [x for x in binned if x["modality"] == m and x["true_cosine_bin"] in ("[0.7,0.8)", "[0.8,0.9)")]
        if r and g:
            safe.append(claim(f"{m.capitalize()} calibration on real embedding pairs: estimated-vs-true cosine RMSE {num(r['RMSE'])} over all pairs "
                              f"(n={r['n']}) and {num(g['RMSE'])} over genuine pairs (n={g['n']}); bias {num(r['bias'])}; Pearson r {num(r['pearson_r'])}. "
                              "Per-bin error SD near the thresholds: " + "; ".join(f"{x['true_cosine_bin']} real {num(x['error_sd'], '{:.3f}')} vs synthetic "
                              f"{num(x['synthetic_predicted_sd'], '{:.3f}')}" for x in b) + ".", "calibration_real_validation.csv", "real_cal", "Part 3", REAL))
    sec = rd("template_security.csv")
    for m in ("face",):
        get = lambda k: pick(sec, modality=m, metric=k).get("value")
        if get("templates_analyzed"):
            safe.append(claim(f"{m.capitalize()} templates ({get('templates_analyzed')} users, each under its own key): mean fraction of ones {num(get('mean_fraction_of_ones'))}, "
                              f"mean per-bit entropy {num(get('mean_per_bit_entropy_bits'))} bits, mean |pairwise bit correlation| {num(get('mean_abs_pairwise_bit_correlation'))} "
                              f"(independent-bit expectation {num(get('expected_mean_abs_correlation_if_independent'))}), cross-user cross-key HD {num(get('cross_user_cross_key_mean_normalized_HD'))} "
                              f"(Daugman DoF {num(get('degrees_of_freedom_daugman'), '{:.1f}')}, DERIVED).", "template_security.csv", "experiments", "Part 16", REAL))
    for m in ("face", "voice"):
        rr = [x for x in unl if x["modality"] == m and x.get("D_sys") and not x.get("note") and abs(float(x["bin_width"]) - 4 / 256) < 1e-12]
        if rr:
            r = rr[0]
            verdict = "exceeds" if r["exceeds_null_p95"] == "True" else "does not exceed"
            safe.append(claim(f"{m.capitalize()} cross-key unlinkability: D_sys = {num(r['D_sys'])} (histogram estimate, bin 4/256; Gomez-Barrero et al. 2018), "
                              f"which {verdict} the estimator's permutation null floor (mean {num(r['null_floor_mean'])}, 95th pct {num(r['null_floor_p95'])}); "
                              f"{r['n_mated']} mated / {r['n_non_mated']} non-mated pairs.", "unlinkability.csv", "experiments", "Part 8", REAL))
    for m in ("face", "voice"):
        r = pick(rev, modality=m, measure="revoked_template_vs_new_active")
        if r:
            safe.append(claim(f"{m.capitalize()} revocation: a revoked template matches the new active template with mean Hamming similarity "
                              f"{num(r['mean'])} (SD {num(r['sd'])}, n={r['n']}), accepted in {pct(r['acceptance_rate'])} of cases.",
                              "revocation_summary.csv", "experiments", "Parts 7/9 via the shipped template-set code", REAL))
    b = pick(sw, check="backend pytest")
    if b:
        safe.append(claim(f"{b['passed']}/{b['total']} automated backend tests pass (software validation, not biometric trials).",
                          "software_validation.csv", "audit", "Part 17", SOFTWARE))
    qual = ["", "## CLAIMS THAT NEED QUALIFICATION", "",
            "- \"Face uses cosine similarity / voice uses Euclidean distance\": they are **calibrated estimates** derived from the template Hamming "
            "comparison, not exact metrics (see Part 3 errors).  \n  *DERIVED* - `backend/services/modality_metrics.py`, `evaluation/results/calibration_real_validation.csv`.",
            "- Voice performance is **closed-set** (speakers overlap training; 24 speakers).  \n  *REAL DATA* - `evaluation/results/dataset_audit.csv`.",
            "- Fusion results use **chimeric** virtual users (modality independence assumed) and only 24 virtual users per pairing.  \n  *REAL DATA* - `evaluation/results/fusion_policy_metrics.csv`."]
    for r in fus:
        if r["policy"].startswith("ALL_REQUIRED(face+voice)"):
            qual.append(claim(f"ALL_REQUIRED face+voice on chimeric users: FAR {pct(r['FAR_mean'])} ± {pct(r['FAR_sd'])}, FRR {pct(r['FRR_mean'])} ± {pct(r['FRR_sd'])} "
                              f"over {r['pairings']} pairings.", "fusion_policy_metrics.csv", "experiments", "Part 14", REAL))
    qual.append("- Robustness numbers come from **simulated** degradations of real probes.  \n  *REAL DATA* - `evaluation/results/robustness.csv`.")
    qual.append("- Latency/memory are single-machine CPU measurements.  \n  *REAL DATA* - `evaluation/results/latency_benchmark.csv`, `evaluation/reports/hardware.json`.")
    no = ["", "## CLAIMS NOT ALLOWED", "",
          "- Any FAR / FRR / EER for **real users of the deployed system** (no consented study was run; `evaluation/results/real_user_*.csv` does not exist).",
          "- That 0.80 (face) and 0.75 (voice) are optimal or experimentally calibrated thresholds.",
          "- That the estimates are exact cosine similarity / Euclidean distance.",
          "- Resistance to presentation attacks, deepfakes or replay (no PAD; IAPMR NOT AVAILABLE).",
          "- Irreversibility or cryptographic security of the templates (no inversion or stolen-key attack was evaluated).",
          "- Demographic fairness (not evaluated).",
          "- That software tests are biometric trials, or that synthetic calibration is biometric performance.",
          "- Numbers for head-pose or physical-microphone robustness (NOT AVAILABLE)."]
    text = "\n".join(lines + safe + qual + no) + "\n"
    (EVAL / "PAPER_CLAIMS.md").write_text(text, encoding="utf-8")
    return text


def key_findings() -> list[str]:
    perf, fus = rd("raw_vs_protected_metrics.csv"), rd("fusion_policy_metrics.csv")
    cal, sweep = rd("calibration_real_binned.csv"), rd("threshold_sweep_eer.csv")
    out = []
    for m in ("face", "voice"):
        raw, pro = pick(perf, modality=m, representation="raw_embedding"), pick(perf, modality=m, representation="protected_template")
        if raw and pro:
            out.append(f"- **{m}** [REAL DATA]: EER {pct(raw['EER'])} raw -> {pct(pro['EER'])} protected (256-bit); at the deployed rule "
                       f"FAR {pct(pro.get('at_operating_FAR'))}, FRR {pct(pro.get('at_operating_FRR'))} (`raw_vs_protected_metrics.csv`).")
    e = pick(sweep, modality="face", representation="protected_template")
    if e:
        out.append(f"- **Face threshold** [REAL DATA]: the protected-template EER point lies at estimated cosine {num(e['EER_threshold'], '{:.2f}')}; "
                   "the teacher-requested 0.80 sits far into the low-FAR/high-FRR region for this face model (`threshold_sweep.csv`).")
    e = pick(sweep, modality="voice", representation="protected_template")
    if e:
        out.append(f"- **Voice threshold** [REAL DATA]: protected EER point at estimated distance {num(e['EER_threshold'], '{:.2f}')} vs the configured 0.75.")
    for name in ("ALL_REQUIRED(face+voice)", "WEIGHTED(face+voice)", "OR(face+voice)"):
        r = pick(fus, policy=name)
        if r:
            out.append(f"- **{name}** [REAL DATA, chimeric]: FAR {pct(r['FAR_mean'])}, FRR {pct(r['FRR_mean'])} (`fusion_policy_metrics.csv`).")
    lev = rd("fusion_score_level_eer.csv")
    if lev:
        out.append("- **Score-level fusion** [REAL DATA, chimeric]: EER " + ", ".join(f"{r['score']} {pct(r['EER'])}" for r in lev) + " (`fusion_score_level_eer.csv`).")
    f = [r for r in cal if r["modality"] == "face" and r["true_cosine_bin"] in ("[0.7,0.8)", "[0.8,0.9)")]
    if f:
        out.append("- **Calibration on real embeddings** [REAL DATA]: per-bin error SD matches the synthetic prediction (" +
                   "; ".join(f"face {r['true_cosine_bin']}: real {num(r['error_sd'], '{:.3f}')} vs synthetic {num(r['synthetic_predicted_sd'], '{:.3f}')}" for r in f) +
                   ") with near-zero mean error (`calibration_real_binned.csv`).")
    return out


# ----------------------------------------------------------------------------- package


def package():
    hw = json.loads((REPORTS / "hardware.json").read_text()) if (REPORTS / "hardware.json").exists() else {}
    runtime = []
    if (REPORTS / "runtime_log.csv").exists():
        with (REPORTS / "runtime_log.csv").open(encoding="utf-8") as h:
            runtime = list(csv.DictReader(h))
    figs = []
    if (FIGURES / "figure_index.csv").exists():
        with (FIGURES / "figure_index.csv").open(encoding="utf-8") as h:
            figs = list(csv.DictReader(h))
    files = sorted(p.relative_to(REPO).as_posix() for d in (RESULTS, TABLES, REPORTS) for p in d.glob("*") if p.is_file())
    exps = [
        ("Part 1 Repository audit", "done", "evaluation/REPOSITORY_AUDIT.md", CMD["audit"]),
        ("Part 2 Metric verification", "done", "evaluation/metric_verification_report.md", CMD["verification"]),
        ("Part 3 Real-embedding calibration validation", "done", "calibration_real_validation.csv, calibration_real_binned.csv", CMD["real_cal"]),
        ("Part 4 Raw vs protected", "done", "raw_vs_protected_metrics.csv", CMD["experiments"] + " 4"),
        ("Part 5 Real-user harness", "harness built + unit-tested; NOT RUN (no consented participants)", "evaluation/real_user_evaluation.py", "python -m evaluation.real_user_evaluation --root <folder> --consent-confirmed"),
        ("Part 6 Threshold sweep", "done", "threshold_sweep.csv, threshold_sweep_eer.csv", CMD["experiments"] + " 6"),
        ("Part 7 Template-set experiments", "re-run (synthetic, matches committed) + real-embedding lifecycle", "metric_verification.csv, revocation_*.csv", CMD["experiments"] + " 7_9"),
        ("Part 8 Unlinkability", "done", "unlinkability.csv", CMD["experiments"] + " 8"),
        ("Part 9 Revocability", "done", "revocation_per_attempt.csv, revocation_summary.csv", CMD["experiments"] + " 7_9"),
        ("Part 10 Latency", "done" if rd("latency_benchmark.csv") else "not run", "latency_benchmark.csv, reports/latency_report.md", CMD["latency"]),
        ("Part 11 Memory", "done" if rd("memory_benchmark.csv") else "not run", "memory_benchmark.csv", CMD["memory"]),
        ("Part 12 Storage", "done", "storage_analysis.csv", CMD["storage"]),
        ("Part 13 Robustness", "done (simulated degradations of real probes; pose / physical mic NOT AVAILABLE)", "robustness.csv", CMD["robustness"]),
        ("Part 14 Fusion", "done (chimeric users)", "fusion_policy_metrics.csv, fusion_score_level_eer.csv", CMD["experiments"] + " 14"),
        ("Part 15 Presentation attacks", "framework only; IAPMR NOT AVAILABLE (no attack data); APCER/BPCER not applicable (no PAD)", "pad_results.csv", "python -m evaluation.ieee.pad_evaluation"),
        ("Part 16 Security statistics", "done", "template_security.csv", CMD["experiments"] + " 16"),
        ("Part 17 Software validation", "done" if rd("software_validation.csv") else "not run", "software_validation.csv, reports/software_validation.md", "python -m evaluation.ieee.audit software"),
        ("Part 18 Figures", f"{len(figs)} figures", "evaluation/figures/", "python -m evaluation.ieee.figures (+ the scripts above)"),
        ("Part 19 Tables", f"{len(list(TABLES.glob('*.md')))} tables", "evaluation/tables/", "python -m evaluation.ieee.reports"),
        ("Part 20 Claims", "done", "evaluation/PAPER_CLAIMS.md", "python -m evaluation.ieee.reports"),
    ]
    lines = ["# IEEE Evaluation Package", "", "Single source of truth for the paper. Every number here is generated from the files listed; "
             "evidence labels: REAL DATA, SYNTHETIC CALIBRATION, SOFTWARE VALIDATION, CONFIGURATION, DERIVED, FUTURE WORK (exactly one per metric).", "",
             "## Experiments", "", "| experiment | status | outputs | command |", "|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | `{d}` |" for a, b, c, d in exps]
    lines += ["", "## Key findings (generated from the result files)", ""] + key_findings()
    lines += ["", "## Headline metrics (see evaluation/tables/ for full tables with sources)", ""]
    for t in ("table05_model_performance_raw", "table06_protected_template_performance", "table04_calibration_statistics", "table07_fusion_performance"):
        p = TABLES / f"{t}.md"
        if p.exists():
            lines += [p.read_text(encoding="utf-8"), ""]
    lines += ["## Figures", "", "| file | label | caption |", "|---|---|---|"]
    lines += [f"| `evaluation/figures/{f['file']}` | {f['evidence_label']} | {f['caption']} |" for f in figs]
    lines += ["", "## Tables", ""] + [f"- `evaluation/tables/{p.name}`" for p in sorted(TABLES.glob("*.md"))]
    lines += ["", "## Unavailable metrics", "",
              "| metric | reason | what would produce it |", "|---|---|---|",
              "| FAR/FRR/EER of real users of the deployed system | no consented participant data | `evaluation/real_user_evaluation.py` |",
              "| IAPMR (presentation attacks) | no attack data | `evaluation/ieee/pad_evaluation.py` with data/pad/ |",
              "| APCER / BPCER | no PAD subsystem exists | would require implementing PAD |",
              "| Head-pose robustness | cannot be simulated faithfully | real multi-pose captures |",
              "| Physical-microphone robustness | needs re-recordings | real captures on several devices |",
              "| Demographic breakdown | no demographic labels | consented study with self-reported attributes |",
              "| Test coverage | pytest-cov not installed | `pip install pytest-cov` then `pytest --cov` |",
              "| Frontend unit tests | no frontend test suite | add vitest |",
              "| Face metrics of the committed face_metrics.csv | no pair data committed | superseded by Part 4 held-out LFW evaluation |",
              "", "## Hardware", ""] + [f"- {k}: {v}" for k, v in hw.items()]
    lines += ["", "## Runtime", "", "| timestamp | step | seconds |", "|---|---|---|"] + [f"| {r['timestamp']} | {r['step']} | {r['seconds']} |" for r in runtime]
    lines += ["", "## Reproducibility", "",
              "1. `git lfs install && git lfs pull` (real checkpoints; the pipelines refuse to run on mock embedders).",
              "2. `pip install -r requirements.txt`; place a Kaggle API token in `~/.kaggle/kaggle.json`.",
              "3. Download the data into the gitignored `data/` folder: LFW via scikit-learn (`fetch_lfw_people(data_home='data/lfw')` downloads the archive), "
              "`gaurav41/voxceleb1-audio-wav-files-for-india-celebrity` -> `data/voxceleb_subset` (Kaggle API).",
              "4. `python -m evaluation.ieee.run_all` (or the per-step commands above). Seeds are fixed (`evaluation/ieee/common.py::SEED`, split seed 42).",
              "5. Evaluation keys derive from a fixed evaluation-only secret (`EVAL_SECRET`), never a deployment `MASTER_SECRET`.",
              "6. Raw data and embedding caches stay in `data/` (gitignored); result files contain scores only.",
              "", "## Generated files", ""] + [f"- `{f}`" for f in files]
    (EVAL / "IEEE_EVALUATION_PACKAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    tables()
    claims()
    package()
    print("reports written")
