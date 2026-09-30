"""Baseline (bounding-box crop) vs aligned (5-landmark similarity) face preprocessing: controlled A/B evaluation.

Systems (checkpoint + preprocessing):
  A  deployed checkpoint        + bbox        (current production)
  B  deployed checkpoint        + aligned     (INFERENCE-TIME PREPROCESSING ABLATION: checkpoint never saw aligned faces)
  C  reproduced bbox checkpoint + bbox        (training/face/runs/face-20260925T031117Z, seed 42)
  D  face_aligned_v1 checkpoint + aligned     (training/face/aligned_v1, seed 42, identical recipe)
  E  face_bbox_fullframe_v2     + bbox        (training/face/bbox_fullframe_v2: full 250x250 LFW frames, seed 42)
  F  face_aligned_v2            + aligned     (training/face/aligned_v2: same full frames, seed 42)
PRIMARY comparison: E vs F (the only difference is the preprocessing used for training AND inference).
C vs D is CONFOUNDED: both trained on scikit-learn's 125x94 LFW slice, which leaves aligned training crops ~39% black
border (vs ~1% at evaluation) - reported, but not used for the decision.

Protocol: held-out LFW identities (2-19 images; disjoint from fine-tuning), the evaluation/ieee enrollment-vs-probe
protocol (first image = reference, others = genuine probes, 50 impostor probes per identity, seed SEED), restricted to
images valid under BOTH preprocessors so every system is scored on IDENTICAL pairs. One MTCNN detection per image
produces both crops. Protected scores use the unchanged 256-bit BioHash (evaluation/ieee/protected.py).

Run: python -m evaluation.ieee.face_alignment_eval [extract|quality|compare|threshold|fusion|robustness|latency|examples ...]
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from evaluation.ieee.common import (
    CACHE, COLORS, COL_W, DBL_W, FIGURES, REAL, REPO, RESULTS, SEED, DERIVED, bootstrap_ci, describe, eer, full_report,
    hardware, ieee_style, rates, roc, save_fig, timed, write_csv,
)
from evaluation.ieee.protected import BITS, build_comparisons, estimated_cosine, hamming_similarity, key_for, templates
from preprocessing.face import ALIGNMENT_TEMPLATE_160, AlignmentFailed, align_face, estimate_similarity_transform, validate_landmarks

CKPT = {
    "deployed": REPO / "models/face/saved/face_embedder.pt",
    "repro_bbox": REPO / "training/face/runs/face-20260925T031117Z/checkpoints/face_embedder.pt",
    "aligned_v1": REPO / "training/face/aligned_v1/runs/face_aligned_v1/checkpoints/face_embedder.pt",
    "bbox_v2": REPO / "training/face/bbox_fullframe_v2/runs/face_bbox_fullframe_v2/checkpoints/face_embedder.pt",
    "aligned_v2": REPO / "training/face/aligned_v2/runs/face_aligned_v2/checkpoints/face_embedder.pt",
}
SYSTEMS = {  # name -> (checkpoint key, preprocessing, description)
    "A_deployed_bbox": ("deployed", "bbox", "deployed checkpoint + bounding-box crop (production)"),
    "B_deployed_aligned": ("deployed", "aligned", "deployed checkpoint + aligned crop (inference-time preprocessing ablation)"),
    "C_repro_bbox": ("repro_bbox", "bbox", "reproduced bbox checkpoint (sklearn 125x94 slice) + bbox crop (v1 pair, confounded)"),
    "D_aligned_v1": ("aligned_v1", "aligned", "face_aligned_v1 (sklearn slice, ~39% black training crops) + aligned crop (v1 pair, confounded)"),
    "E_bbox_v2": ("bbox_v2", "bbox", "face_bbox_fullframe_v2 checkpoint + bounding-box crop (PRIMARY baseline)"),
    "F_aligned_v2": ("aligned_v2", "aligned", "face_aligned_v2 checkpoint + aligned crop (PRIMARY aligned)"),
}
BASE, ALIGNED = "E_bbox_v2", "F_aligned_v2"
CACHE_FILE = CACHE / "face_alignment_embeddings.npz"
OUT_FIG = FIGURES / "face_alignment_examples"
THRESH = 0.80  # production face rule: estimated cosine >= 0.80 (not changed by this experiment)


# ----------------------------------------------------------------------------- detection + both crops


def _detector():
    from facenet_pytorch import MTCNN

    return MTCNN(image_size=160, margin=0, post_process=True)


def detect_both(rgb: np.ndarray, detector):
    """(bbox_crop, aligned_crop, box, landmarks) from ONE detection of the largest face, reproducing
    FacePreprocessor.preprocess for the bbox crop and FacePreprocessorAligned.preprocess for the aligned crop."""
    from facenet_pytorch import extract_face, fixed_image_standardization
    from PIL import Image

    pil = Image.fromarray(rgb)
    boxes, _, points = detector.detect(pil, landmarks=True)
    if boxes is None or len(boxes) == 0:
        raise ValueError("no face")
    boxes = np.asarray(boxes, dtype=np.float64)
    k = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    t = fixed_image_standardization(extract_face(pil, boxes[k], image_size=160, margin=0))
    bbox_crop = ((t.permute(1, 2, 0).numpy() * 128.0) + 127.5).clip(0, 255).astype(np.uint8)
    try:
        aligned = align_face(rgb, points[k])
        lm = validate_landmarks(points[k])
    except AlignmentFailed:
        aligned, lm = None, None
    return bbox_crop, aligned, boxes[k], lm


def _rgb(path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def _embedders():
    from models.face.inference import FaceEmbedder

    out = {}
    for key, path in CKPT.items():
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint {path} - train it first (training/run_training.py)")
        out[key] = FaceEmbedder(checkpoint_path=path)
        assert not out[key].mock_mode
    return out


def extract():
    """Embeddings of every held-out LFW image under all four systems + per-image geometry (cached under data/)."""
    from preprocessing.face import FacePreprocessor, FacePreprocessorAligned

    base = np.load(CACHE / "face_embeddings.npz", allow_pickle=True)
    paths, labels = base["paths"], base["labels"]
    det, emb = _detector(), _embedders()
    # faithfulness check: our single-detection crops == the real preprocessors' outputs
    fb, fa = FacePreprocessor(), FacePreprocessorAligned()
    for p in paths[:40]:
        img = _rgb(p)
        b, a, _, _ = detect_both(img, det)
        assert np.array_equal(b, fb.preprocess(img)), "bbox crop differs from FacePreprocessor.preprocess"
        assert np.array_equal(a, fa.preprocess(img)), "aligned crop differs from FacePreprocessorAligned.preprocess"
    E = {s: np.zeros((len(paths), 512), np.float32) for s in SYSTEMS}
    ok_bbox = np.zeros(len(paths), bool)
    ok_aligned = np.zeros(len(paths), bool)
    geo = np.full((len(paths), 5, 2), np.nan)  # landmarks in baseline-crop coordinates (transient, cache only)
    geo_after = np.full((len(paths), 5, 2), np.nan)
    t0 = time.perf_counter()
    for i, p in enumerate(paths):
        img = _rgb(p)
        try:
            b, a, box, lm = detect_both(img, det)
        except ValueError:
            continue
        ok_bbox[i] = True
        crops = {"bbox": b}
        if a is not None:
            ok_aligned[i] = True
            crops["aligned"] = a
            x1, y1, x2, y2 = box
            geo[i] = np.c_[(lm[:, 0] - x1) * 160 / (x2 - x1), (lm[:, 1] - y1) * 160 / (y2 - y1)]
            M = estimate_similarity_transform(lm, ALIGNMENT_TEMPLATE_160)
            geo_after[i] = lm @ M[:, :2].T + M[:, 2]
        for s, (ck, pre, _) in SYSTEMS.items():
            if pre in crops:
                E[s][i] = emb[ck].extract_embedding(crops[pre])
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(paths)} {(i + 1) / (time.perf_counter() - t0):.1f}/s", flush=True)
    np.savez_compressed(CACHE_FILE, labels=labels, paths=paths, ok_bbox=ok_bbox, ok_aligned=ok_aligned, geo=geo, geo_after=geo_after,
                        seconds=time.perf_counter() - t0, **{f"E_{s}": v for s, v in E.items()})


def _load():
    z = np.load(CACHE_FILE, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _comparisons(z):
    """Identical pairs for every system: build on images valid under BOTH preprocessors, then re-score."""
    valid = z["ok_bbox"] & z["ok_aligned"]
    idx = np.nonzero(valid)[0]
    ref = build_comparisons("face", z[f"E_{BASE}"][idx], z["labels"][idx], impostors_per_identity=50)
    out = {}
    for s in SYSTEMS:
        c = build_comparisons("face", z[f"E_{s}"][idx], z["labels"][idx], impostors_per_identity=50)
        assert np.array_equal(c.ref_index, ref.ref_index) and np.array_equal(c.probe_index, ref.probe_index), "pairs differ"
        out[s] = c
    return idx, out


# ----------------------------------------------------------------------------- Part 8: geometric normalization


def quality():
    z = _load()
    ok = z["ok_aligned"]
    before, after = z["geo"][ok], z["geo_after"][ok]
    T = ALIGNMENT_TEMPLATE_160

    def metrics(P):
        le, re_ = P[:, 0], P[:, 1]
        eye_mid = (le + re_) / 2
        mouth_mid = (P[:, 3] + P[:, 4]) / 2
        return {
            "eye_line_angle_abs_deg": np.abs(np.degrees(np.arctan2(re_[:, 1] - le[:, 1], re_[:, 0] - le[:, 0]))),
            "inter_eye_distance_px": np.linalg.norm(re_ - le, axis=1),
            "eye_midpoint_offset_from_center_px": np.linalg.norm(eye_mid - np.array([80.0, 80.0]), axis=1),
            "mouth_midpoint_y_px": mouth_mid[:, 1],
            "landmark_rms_error_to_template_px": np.sqrt(((P - T) ** 2).sum(axis=2).mean(axis=1)),
        }

    mb, ma = metrics(before), metrics(after)
    rows = []
    for name in mb:
        for stage, m in (("before_alignment (baseline bbox crop)", mb), ("after_alignment", ma)):
            v = m[name]
            rows.append({"metric": name, "stage": stage, "n": len(v), "mean": v.mean(), "median": np.median(v), "sd": v.std(ddof=1),
                         "p95": np.percentile(v, 95)})
    rows.append({"metric": "failure_to_align", "stage": "aligned pipeline", "n": int(z["ok_bbox"].sum()),
                 "mean": float((z["ok_bbox"] & ~z["ok_aligned"]).sum() / z["ok_bbox"].sum())})
    rows.append({"metric": "no_face_detected", "stage": "both pipelines", "n": len(z["ok_bbox"]), "mean": float((~z["ok_bbox"]).mean())})
    write_csv(RESULTS / "face_alignment_quality.csv", rows, REAL)
    plt = ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 2.0))
    for ax, name, bins in ((axes[0], "eye_line_angle_abs_deg", np.linspace(0, 25, 51)),
                           (axes[1], "inter_eye_distance_px", np.linspace(40, 110, 71)),
                           (axes[2], "landmark_rms_error_to_template_px", np.linspace(0, 35, 71))):
        ax.hist(mb[name], bins=bins, alpha=0.6, color=COLORS["raw"], label="baseline bbox crop")
        ax.hist(ma[name], bins=bins, alpha=0.6, color=COLORS["face"], label="aligned")
        ax.set(xlabel=name.replace("_", " "), ylabel="images")
    axes[0].legend(fontsize=6)
    save_fig(fig, "face_alignment_quality", REAL,
             "Geometric variation of the 5 landmarks in the 160x160 face crop, baseline bbox crop vs similarity-aligned (held-out LFW).")


# ----------------------------------------------------------------------------- Part 9/14: raw + protected comparison


def _scores(c):
    s = {"raw_cosine": c.raw_cosine, "hamming_similarity": c.hamming, "estimated_cosine": estimated_cosine(c.hamming, "face")}
    return s


def compare():
    z = _load()
    idx, comps = _comparisons(z)
    rows = []
    for sname, c in comps.items():
        for rep, key, thr in (("raw_embedding", "raw_cosine", THRESH), ("protected_template_256bit", "estimated_cosine", THRESH)):
            s = _scores(c)[key]
            g, i = s[c.genuine], s[~c.genuine]
            r = full_report(g, i, True, thr)
            lo, hi = bootstrap_ci(g, i, lambda a, b: eer(a, b)[0], n_boot=100)
            h = c.hamming
            rows.append({"system": sname, "description": SYSTEMS[sname][2], "representation": rep, "score": key, **r,
                         "EER_ci95_low": lo, "EER_ci95_high": hi,
                         "genuine_score_mean": g.mean(), "genuine_score_sd": g.std(ddof=1), "impostor_score_mean": i.mean(), "impostor_score_sd": i.std(ddof=1),
                         "genuine_hamming_mean": h[c.genuine].mean(), "impostor_hamming_mean": h[~c.genuine].mean()})
    meta = {"identities": int(len(set(z["labels"][idx].tolist()))), "images_valid_both": int(len(idx)), "images_total": int(len(z["labels"])),
            "genuine_pairs": int(comps[BASE].genuine.sum()), "impostor_pairs": int((~comps[BASE].genuine).sum()),
            "impostors_per_identity": 50, "seed": SEED, "identity_overlap_with_training": "none (LFW identities with 2-19 images; training used >= 20)"}
    for r in rows:
        r.update({f"protocol_{k}": v for k, v in meta.items()})
    write_csv(RESULTS / "face_alignment_metrics.csv", rows, REAL)
    # per-pair file (scores only)
    C, D = comps[BASE], comps[ALIGNED]
    sc = {k: _scores(v) for k, v in comps.items()}
    pair_rows = []
    for j in range(len(C.genuine)):
        pair_rows.append({
            "pair_id": j, "sample_id": int(idx[C.probe_index[j]]), "reference_id": int(idx[C.ref_index[j]]), "genuine": bool(C.genuine[j]),
            "baseline_score": round(float(sc[BASE]["estimated_cosine"][j]), 5), "aligned_score": round(float(sc[ALIGNED]["estimated_cosine"][j]), 5),
            "baseline_prediction": bool(sc[BASE]["estimated_cosine"][j] >= THRESH), "aligned_prediction": bool(sc[ALIGNED]["estimated_cosine"][j] >= THRESH),
            "baseline_raw_cosine": round(float(C.raw_cosine[j]), 5), "aligned_raw_cosine": round(float(D.raw_cosine[j]), 5),
            "ablation_deployed_bbox_score": round(float(sc["A_deployed_bbox"]["estimated_cosine"][j]), 5),
            "ablation_deployed_aligned_score": round(float(sc["B_deployed_aligned"]["estimated_cosine"][j]), 5),
            "v1_confounded_bbox_score": round(float(sc["C_repro_bbox"]["estimated_cosine"][j]), 5),
            "v1_confounded_aligned_score": round(float(sc["D_aligned_v1"]["estimated_cosine"][j]), 5),
        })
    write_csv(RESULTS / "face_alignment_comparison.csv", pair_rows, REAL)
    _plot_distributions(comps)


def _plot_distributions(comps):
    plt = ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W * 0.75, 2.1), sharey=True)
    for ax, s, title in ((axes[0], BASE, "baseline (bbox)"), (axes[1], ALIGNED, "aligned")):
        c = comps[s]
        e = estimated_cosine(c.hamming, "face")
        bins = np.linspace(-0.25, 1.0, 60)
        ax.hist(e[~c.genuine], bins=bins, density=True, alpha=0.55, color=COLORS["raw"], label="impostor")
        ax.hist(e[c.genuine], bins=bins, density=True, alpha=0.55, color=COLORS["face"], label="genuine")
        ax.axvline(THRESH, color="k", ls="--", lw=0.7)
        ax.set(xlabel="estimated cosine (256-bit template)", title=title)
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=6)
    save_fig(fig, "face_alignment_score_distributions", REAL, "Genuine/impostor protected-template scores, baseline vs aligned face model (identical pairs).")


def paired(n_boot: int = 300):
    """Paired bootstrap of the protected-template EER difference on IDENTICAL pairs, resampling enrolled identities."""
    z = _load()
    _, comps = _comparisons(z)
    contrasts = {"F_aligned_v2 - E_bbox_v2 (controlled, PRIMARY)": (BASE, ALIGNED),
                 "F_aligned_v2 - A_deployed_bbox": ("A_deployed_bbox", ALIGNED),
                 "B_deployed_aligned - A_deployed_bbox (ablation)": ("A_deployed_bbox", "B_deployed_aligned"),
                 "D_aligned_v1 - C_repro_bbox (confounded)": ("C_repro_bbox", "D_aligned_v1")}
    rows = []
    for name, (a, b) in contrasts.items():
        ca, cb = comps[a], comps[b]
        sa, sb = estimated_cosine(ca.hamming, "face"), estimated_cosine(cb.hamming, "face")
        g = ca.genuine
        ids = np.array(sorted(set(ca.target.tolist())), dtype=object)
        groups = {t: np.nonzero(ca.target == t)[0] for t in ids}
        rng = np.random.default_rng(0)
        d = []
        for _ in range(n_boot):
            pick = np.concatenate([groups[t] for t in rng.choice(ids, len(ids))])
            gg = g[pick]
            d.append(eer(sb[pick][gg], sb[pick][~gg])[0] - eer(sa[pick][gg], sa[pick][~gg])[0])
        d = np.array(d)
        rows.append({"contrast": name, "system_a": a, "system_b": b, "EER_a": eer(sa[g], sa[~g])[0], "EER_b": eer(sb[g], sb[~g])[0],
                     "delta_EER_b_minus_a": eer(sb[g], sb[~g])[0] - eer(sa[g], sa[~g])[0],
                     "delta_ci95_low": np.percentile(d, 2.5), "delta_ci95_high": np.percentile(d, 97.5),
                     "fraction_resamples_delta_ge_0": float((d >= 0).mean()), "n_boot": n_boot, "resampling_unit": "enrolled identity",
                     "n_identities": len(ids), "n_genuine": int(g.sum()), "n_impostor": int((~g).sum())})
    write_csv(RESULTS / "face_alignment_paired_bootstrap.csv", rows, REAL)


# ----------------------------------------------------------------------------- Part 15: thresholds


def threshold():
    z = _load()
    _, comps = _comparisons(z)
    rows = []
    grid = np.round(np.arange(0.60, 0.9001, 0.02), 2)
    for s, c in comps.items():
        for rep, key in (("protected_estimated_cosine", "estimated_cosine"), ("raw_cosine", "raw_cosine")):
            v = _scores(c)[key]
            g, i = v[c.genuine], v[~c.genuine]
            e, t = eer(g, i)
            for thr in grid:
                r = rates(g, i, float(thr))
                rows.append({"system": s, "score": rep, "threshold": thr, "FAR": r["FAR"], "FRR": r["FRR"], "TAR": r["TAR"],
                             "system_EER": e, "system_EER_threshold": t})
    write_csv(RESULTS / "face_threshold_comparison.csv", rows, REAL)
    plt = ieee_style()
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for s, color in ((BASE, COLORS["raw"]), (ALIGNED, COLORS["face"])):
        rr = [r for r in rows if r["system"] == s and r["score"] == "protected_estimated_cosine"]
        ax.plot([r["threshold"] for r in rr], [r["FRR"] for r in rr], color=color, label=f"FRR {s}")
        ax.plot([r["threshold"] for r in rr], [r["FAR"] for r in rr], "--", color=color, label=f"FAR {s}")
    ax.axvline(THRESH, color="k", lw=0.6, ls=":")
    ax.set(xlabel="estimated cosine threshold (accept >=)", ylabel="rate", ylim=(0, 1), title="Face threshold sweep (protected)")
    ax.grid(True)
    ax.legend(fontsize=5.5)
    save_fig(fig, "face_threshold_comparison", REAL, "FAR/FRR vs face threshold, baseline vs aligned (256-bit protected templates, identical pairs).")


# ----------------------------------------------------------------------------- Part 16: fusion (face + voice)


def fusion(pairings: int = 20, attempts: int = 3):
    from backend.config import Settings
    from backend.services.modality_metrics import decide
    from fusion.config import FusionPolicy
    from fusion.policy import evaluate_fusion_policy

    z = _load()
    valid = z["ok_bbox"] & z["ok_aligned"]
    labels = z["labels"]
    face_ids = {}
    for i in np.nonzero(valid)[0]:
        face_ids.setdefault(labels[i], []).append(i)
    face_ids = {k: v for k, v in face_ids.items() if len(v) >= attempts + 1}
    v = np.load(CACHE / "voice_embeddings.npz", allow_pickle=True)
    VE, VL = v["embeddings"], v["labels"]
    speakers = sorted(set(VL.tolist()))
    vpool = {s: list(np.nonzero(VL == s)[0]) for s in speakers}
    settings = Settings(master_secret="x")
    rng = np.random.default_rng(SEED)
    rows = []
    for p in range(pairings):
        faces = rng.choice(sorted(face_ids), len(speakers), replace=False)  # same face identities for every system
        for s in SYSTEMS:
            FE = z[f"E_{s}"]
            for t, (fid, spk) in enumerate(zip(faces, speakers)):
                fk, vk = key_for(f"{p}-{t}", "face"), key_for(f"{p}-{t}", "voice")
                f_idx = [face_ids[fid][0]] + [face_ids[faces[u]][1 + k] for u in range(len(speakers)) for k in range(attempts)]
                v_idx = [vpool[spk][0]] + [vpool[speakers[u]][1 + k] for u in range(len(speakers)) for k in range(attempts)]
                Tf, Tv = templates(FE[f_idx], fk), templates(VE[v_idx], vk)
                hf, hv = hamming_similarity(Tf[1:], Tf[0][None]), hamming_similarity(Tv[1:], Tv[0][None])
                for j in range(len(hf)):
                    u = j // attempts
                    df, dv = decide("face", float(hf[j]), BITS, settings), decide("voice", float(hv[j]), BITS, settings)
                    both = {"face": df, "voice": dv}
                    all_req = evaluate_fusion_policy({m: d.fusion_score for m, d in both.items()}, {m: d.matched for m, d in both.items()},
                                                     FusionPolicy.ALL_REQUIRED, float(np.mean([d.fusion_threshold for d in both.values()])))
                    weighted = evaluate_fusion_policy({m: d.fusion_score for m, d in both.items()}, {m: d.matched for m, d in both.items()},
                                                      FusionPolicy.WEIGHTED, float(np.mean([d.fusion_threshold for d in both.values()])))
                    rows.append((p, s, u == t, df.matched, dv.matched, all_req.authenticated, weighted.authenticated,
                                 (df.fusion_score + dv.fusion_score) / 2, df.fusion_score))
    out = []
    for s in SYSTEMS:
        rr = [r for r in rows if r[1] == s]
        gen = np.array([r[2] for r in rr])
        for name, col in (("face_only", 3), ("face+voice ALL_REQUIRED", 5), ("face+voice WEIGHTED", 6)):
            acc = np.array([r[col] for r in rr])
            out.append({"system": s, "policy": name, "genuine_acceptance": acc[gen].mean(), "impostor_acceptance": acc[~gen].mean(),
                        "FAR": acc[~gen].mean(), "FRR": 1 - acc[gen].mean(), "n_genuine": int(gen.sum()), "n_impostor": int((~gen).sum()),
                        "genuine_rejected_by_face": float(np.mean([not r[3] for r in rr if r[2]])),
                        "genuine_rejected_by_voice": float(np.mean([not r[4] for r in rr if r[2]]))})
        mean_score = np.array([r[7] for r in rr])
        face_score = np.array([r[8] for r in rr])
        out.append({"system": s, "policy": "score-level mean(face,voice)", "EER": eer(mean_score[gen], mean_score[~gen])[0],
                    "face_only_EER": eer(face_score[gen], face_score[~gen])[0], "n_genuine": int(gen.sum()), "n_impostor": int((~gen).sum())})
    write_csv(RESULTS / "face_alignment_fusion.csv", out, REAL)


# ----------------------------------------------------------------------------- Part 17: robustness


def robustness(n_per_class: int = 200):
    from evaluation.ieee.robustness import _gamma, _rotate, _scale

    z = _load()
    idx, comps = _comparisons(z)
    C = comps[BASE]
    rng = np.random.default_rng(SEED)
    sel = np.r_[rng.choice(np.nonzero(C.genuine)[0], n_per_class, replace=False), rng.choice(np.nonzero(~C.genuine)[0], n_per_class, replace=False)]
    conds = {"clean": lambda x: x, "rotation_15deg": lambda x: _rotate(x, 15), "rotation_30deg": lambda x: _rotate(x, 30),
             "dark_gamma2.2": lambda x: _gamma(x, 2.2), "bright_gamma0.45": lambda x: _gamma(x, 0.45),
             "distance_scale0.5": lambda x: _scale(x, 0.5), "distance_scale0.25": lambda x: _scale(x, 0.25),
             "blur_sigma2": lambda x: cv2.GaussianBlur(x, (0, 0), 2)}
    det, emb = _detector(), _embedders()
    rows = []
    for cname, fn in conds.items():
        for s, (ck, pre, _) in ((BASE, SYSTEMS[BASE]), (ALIGNED, SYSTEMS[ALIGNED])):
            E = z[f"E_{s}"][idx]
            h = np.full(len(sel), np.nan)
            for j, k in enumerate(sel):
                img = fn(_rgb(z["paths"][idx[C.probe_index[k]]]))
                try:
                    b, a, _, _ = detect_both(img, det)
                except ValueError:
                    continue
                crop = b if pre == "bbox" else a
                if crop is None:
                    continue
                probe = emb[ck].extract_embedding(crop)
                T = templates(np.vstack([E[C.ref_index[k]], probe]), key_for(C.target[k], "face"))
                h[j] = hamming_similarity(T[1], T[0])
            ok = ~np.isnan(h)
            e = np.where(ok, estimated_cosine(np.nan_to_num(h), "face"), -np.inf)  # failure to acquire = reject
            gen = C.genuine[sel]
            r = rates(e[gen], e[~gen], THRESH)
            rows.append({"condition": cname, "system": s, "failure_to_acquire": float(1 - ok.mean()), "FRR_at_0.80": r["FRR"], "FAR_at_0.80": r["FAR"],
                         "EER": eer(e[gen & ok], e[~gen & ok])[0], "mean_genuine_estimated_cosine": float(e[gen & ok].mean()), "n_genuine": int(gen.sum()), "n_impostor": int((~gen).sum())})
            print(rows[-1], flush=True)
    write_csv(RESULTS / "face_alignment_robustness.csv", rows, REAL)


# ----------------------------------------------------------------------------- Part 18: latency


def _cold_worker(mode: str):
    t0 = time.perf_counter()
    from preprocessing.face import FacePreprocessor, FacePreprocessorAligned

    img = _rgb(np.load(CACHE / "face_embeddings.npz", allow_pickle=True)["paths"][0])
    pre = FacePreprocessorAligned() if mode == "aligned" else FacePreprocessor()
    pre.preprocess(img)
    print(json.dumps({"total_ms": (time.perf_counter() - t0) * 1000}))


def latency(warm: int = 100, cold: int = 10):
    from preprocessing.face import FacePreprocessor, FacePreprocessorAligned

    paths = np.load(CACHE / "face_embeddings.npz", allow_pickle=True)["paths"]
    img = _rgb(paths[0])
    pres = {"baseline_bbox": FacePreprocessor(), "aligned": FacePreprocessorAligned()}
    for p in pres.values():
        p.preprocess(img)  # warm-up
    times = {k: [] for k in pres}
    for _ in range(warm):  # interleaved so background load affects both equally
        for k, p in pres.items():
            t = time.perf_counter()
            p.preprocess(img)
            times[k].append((time.perf_counter() - t) * 1000)
    colds = {k: [] for k in pres}
    for _ in range(cold):
        for k, mode in (("baseline_bbox", "bbox"), ("aligned", "aligned")):
            out = subprocess.run([sys.executable, "-m", "evaluation.ieee.face_alignment_eval", "--cold-worker", mode], cwd=REPO,
                                 capture_output=True, text=True, timeout=600)
            colds[k].append(json.loads(out.stdout.strip().splitlines()[-1])["total_ms"])
    hw = hardware()
    import psutil

    load = psutil.cpu_percent(interval=1.0)
    rows = []
    for k in pres:
        for mode, v in (("warm_preprocess", times[k]), ("cold_import_load_first_preprocess", colds[k])):
            a = np.array(v)
            rows.append({"pipeline": k, "mode": mode, "runs": len(a), "mean_ms": a.mean(), "median_ms": np.median(a), "sd_ms": a.std(ddof=1),
                         "p95_ms": np.percentile(a, 95), "p99_ms": np.percentile(a, 99), "min_ms": a.min(), "max_ms": a.max(), "cpu": hw.get("processor"), "ram_gb": hw.get("ram_gb"),
                         "os": hw.get("os"), "python": hw.get("python"), "pytorch": hw.get("torch"), "gpu": hw.get("gpu"),
                         "background_cpu_load_percent_at_end": load, "note": "runs interleaved (baseline, aligned) so background load affects both equally"})
    write_csv(RESULTS / "face_alignment_latency.csv", rows, REAL)


# ----------------------------------------------------------------------------- Part 7: visual examples


def examples():
    z = _load()
    ok = np.nonzero(z["ok_aligned"])[0]
    geo = z["geo"][ok]
    le, re_, nose = geo[:, 0], geo[:, 1], geo[:, 2]
    ied = np.linalg.norm(re_ - le, axis=1)
    yaw = (nose[:, 0] - (le[:, 0] + re_[:, 0]) / 2) / ied
    pitch = (nose[:, 1] - (le[:, 1] + re_[:, 1]) / 2) / ied
    roll = np.abs(np.degrees(np.arctan2(re_[:, 1] - le[:, 1], re_[:, 0] - le[:, 0])))
    rms = np.sqrt(((geo - ALIGNMENT_TEMPLATE_160) ** 2).sum(axis=2).mean(axis=1))
    picks = {"frontal": np.argmin(np.abs(yaw) + np.abs(roll) / 30), "turned_left": np.argmin(yaw), "turned_right": np.argmax(yaw),
             "looking_up": np.argmin(pitch), "looking_down": np.argmax(pitch), "largest_roll": np.argmax(roll),
             "small_face_in_crop": np.argmin(ied), "large_face_in_crop": np.argmax(ied), "largest_misalignment": np.argmax(rms),
             "median_case": np.argsort(rms)[len(rms) // 2], "borderline_blur": None, "second_largest_roll": np.argsort(roll)[-2]}
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    det = _detector()
    sharp = []
    plt = ieee_style()
    for name, j in picks.items():
        if j is None:
            continue
        i = ok[j]
        img = _rgb(z["paths"][i])
        b, a, box, lm = detect_both(img, det)
        sharp.append(cv2.Laplacian(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
        M = estimate_similarity_transform(lm, ALIGNMENT_TEMPLATE_160)
        lm_after = lm @ M[:, :2].T + M[:, 2]
        fig, axes = plt.subplots(1, 4, figsize=(DBL_W, 1.9))
        axes[0].imshow(img)
        x1, y1, x2, y2 = box
        axes[0].add_patch(__import__("matplotlib").patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=0.8))
        axes[0].set_title("input + MTCNN box")
        axes[1].imshow(img)
        axes[1].scatter(lm[:, 0], lm[:, 1], s=8, c=["r", "r", "y", "c", "c"])
        axes[1].set_title("5 landmarks")
        axes[2].imshow(b)
        axes[2].set_title("baseline bbox crop")
        axes[3].imshow(a)
        axes[3].scatter(ALIGNMENT_TEMPLATE_160[:, 0], ALIGNMENT_TEMPLATE_160[:, 1], s=14, marker="x", c="w", lw=0.8, label="template")
        axes[3].scatter(lm_after[:, 0], lm_after[:, 1], s=6, c="r", label="aligned landmarks")
        axes[3].set_title("aligned 160x160")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{name.replace('_', ' ')} (LFW public dataset; RMS to template {rms[j]:.1f}px before alignment)", fontsize=7)
        fig.savefig(OUT_FIG / f"{name}.png")
        plt.close(fig)
    print("examples written:", sorted(p.name for p in OUT_FIG.glob("*.png")))


STEPS = {"extract": extract, "quality": quality, "compare": compare, "paired": paired, "threshold": threshold, "fusion": fusion,
         "robustness": robustness, "latency": latency, "examples": examples}

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--cold-worker":
        _cold_worker(sys.argv[2])
    else:
        for step in sys.argv[1:] or list(STEPS):
            with timed(f"face_alignment_{step}"):
                STEPS[step]()
