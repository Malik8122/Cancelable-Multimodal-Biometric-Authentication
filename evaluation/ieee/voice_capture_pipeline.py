"""Voice capture robustness: quality gate calibration, speech-duration sensitivity, two-utterance aggregation, latency.

A NEW, separately labelled evaluation. It never rewrites the published voice results (voice_metrics.csv,
protected_vs_raw.csv, robustness.csv): it writes evaluation/results/voice_capture_pipeline.csv only.

Data: the VoxCeleb1 Indian-celebrity TEST split used everywhere else (evaluation/ieee/extract_embeddings.py, 24 speakers,
751 clips; closed-set). Protected domain exactly as deployed: 256-bit BioHash under the target speaker's key, calibrated
estimated Euclidean distance, operating rule `estimated_distance <= 0.75` (CONFIGURATION, teacher-requested - not tuned).

1. gate    - estimated SNR (preprocessing/voice.py::VoicePreprocessor.quality_metrics) on clean clips and with white noise
             added at 20 / 10 / 5 dB (evaluation/ieee/robustness.py::_noise). For each candidate gate threshold: the share
             of recordings rejected, and FRR / FAR at 0.75 over ALL probes vs over the ACCEPTED probes only. Capture
             rejections are reported separately - never mixed into FRR.
2. duration - genuine/impostor probes cut to their first N seconds: FRR / FAR / EER at 0.75 per N (evidence for the
             minimum-speech gate).
3. aggregation - enrollment from 1 vs 2 utterances and probes of 1 vs 2 utterances, where 2 = centroid of the two unit
             embeddings (embeddings/centroid.py, the same rule as face enrollment). Same speakers, same utterances.
4. latency - quality metrics, embedding extraction and the full protected-domain decision for 1 vs 2 utterances.

Run: python -m evaluation.ieee.voice_capture_pipeline
"""

from __future__ import annotations

import time

import numpy as np

from embeddings.centroid import centroid_embedding
from evaluation.ieee.common import CACHE, REAL, RESULTS, SEED, SOFTWARE, eer, rates, write_csv
from evaluation.ieee.experiments import load
from evaluation.ieee.protected import estimated_distance, hamming_similarity, key_for, templates
from evaluation.ieee.robustness import _noise

THRESHOLD = 0.75  # voice estimated Euclidean distance, lower = better (CONFIGURATION)
N_PER_CLASS = 200
GATE_CANDIDATES_DB = (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0)
DURATIONS_S = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


def _distances(E_ref: np.ndarray, probes: np.ndarray, target: str) -> np.ndarray:
    """Estimated distances of `probes` to the reference, all templated under the TARGET's key (one projection build)."""
    T = templates(np.vstack([E_ref[None, :], np.atleast_2d(probes)]), key_for(target, "voice"))
    return estimated_distance(hamming_similarity(T[1:], T[0][None, :]), "voice")


def _distance(E_ref: np.ndarray, E_probe: np.ndarray, target: str) -> float:
    return float(_distances(E_ref, E_probe, target)[0])


def _probe_distances(E, refs, probes, probe_embs) -> np.ndarray:
    out = np.empty(len(probes))
    for target in sorted({t for _, t, _ in probes}):
        rows = [k for k, (_, t, _) in enumerate(probes) if t == target]
        out[rows] = _distances(E[refs[target]], np.stack([probe_embs[k] for k in rows]), target)
    return out


def _summary(gen: np.ndarray, imp: np.ndarray) -> dict:
    if len(gen) == 0 or len(imp) == 0:  # e.g. a strict gate rejected every probe of a condition
        return {"n_genuine": int(len(gen)), "n_impostor": int(len(imp)), "FRR_at_0.75": float("nan"),
                "FAR_at_0.75": float("nan"), "EER": float("nan")}
    r = rates(gen, imp, THRESHOLD, False)
    e = eer(gen, imp, False)[0] if len(gen) and len(imp) else float("nan")
    return {"n_genuine": int(len(gen)), "n_impostor": int(len(imp)), "FRR_at_0.75": r["FRR"], "FAR_at_0.75": r["FAR"], "EER": e}


def _probe_subset(labels: np.ndarray, rng) -> tuple[dict, list[tuple[int, str, bool]]]:
    """Reference = first clip of each speaker (the published protocol); N genuine + N impostor probes sampled."""
    refs = {s: int(np.nonzero(labels == s)[0][0]) for s in sorted(set(labels.tolist()))}
    genuine = [(i, labels[i], True) for i in range(len(labels)) if i != refs[labels[i]]]
    impostor = [(i, t, False) for i in range(len(labels)) for t in refs if t != labels[i]]
    pick = lambda xs: [xs[k] for k in rng.choice(len(xs), min(N_PER_CLASS, len(xs)), replace=False)]
    return refs, pick(genuine) + pick(impostor)


def run_gate_and_duration(rows: list[dict]) -> None:
    from embeddings.pipelines import VoicePipeline
    from preprocessing.voice import VoicePreprocessor, load_wav_file

    d = load("voice")
    E, labels, paths = d["embeddings"], d["labels"], d["paths"]
    rng = np.random.default_rng(SEED)
    refs, probes = _probe_subset(labels, rng)
    pipe, pre = VoicePipeline(), VoicePreprocessor()
    assert not pipe.is_mock, "voice checkpoint missing"

    conditions = {"clean": None, "white_noise_snr20": 20, "white_noise_snr10": 10, "white_noise_snr5": 5}
    cache = CACHE / "voice_capture_pipeline_probes.npz"
    stored = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    for name, snr in conditions.items():
        if f"{name}_dist" in stored:
            dist, snr_est = stored[f"{name}_dist"], stored[f"{name}_snr"]
            gen = np.array([g for _, _, g in probes])
            _gate_rows(rows, name, dist, snr_est, gen)
            continue
        embs, snr_est = [], []
        for idx, target, is_gen in probes:
            w, sr = load_wav_file(paths[idx])
            x = np.asarray(w, float) if snr is None else _noise(np.asarray(w, float), snr, rng)
            x = np.asarray(x, np.float32)
            snr_est.append(pre.quality_metrics(x, sr)["estimated_snr_db"])
            embs.append(pipe.embed(x, sample_rate=sr))
        dist, snr_est = _probe_distances(E, refs, probes, embs), np.array(snr_est)
        gen = np.array([g for _, _, g in probes])
        stored[f"{name}_dist"], stored[f"{name}_snr"] = dist, snr_est
        np.savez(cache, **stored)
        _gate_rows(rows, name, dist, snr_est, gen)

    for seconds in DURATIONS_S:
        key = f"first_{seconds:g}s_dist"
        if key in stored:
            dist = stored[key]
        else:
            embs = []
            for idx, target, is_gen in probes:
                w, sr = load_wav_file(paths[idx])
                embs.append(pipe.embed(np.asarray(w, np.float32)[: int(seconds * sr)], sample_rate=sr))
            dist = stored[key] = _probe_distances(E, refs, probes, embs)
            np.savez(cache, **stored)
        gen = np.array([g for _, _, g in probes])
        rows.append({"evidence_label": REAL, "experiment": "duration", "condition": f"first_{seconds:g}s",
                     **_summary(dist[gen], dist[~gen])})
        print(rows[-1], flush=True)


def _gate_rows(rows, name, dist, snr_est, gen) -> None:
    rows.append({"evidence_label": REAL, "experiment": "gate", "condition": name, "gate_min_snr_db": "none",
                 "rejected_by_gate": 0.0, **_summary(dist[gen], dist[~gen])})
    for t in GATE_CANDIDATES_DB:
        ok = snr_est >= t
        rows.append({"evidence_label": REAL, "experiment": "gate", "condition": name, "gate_min_snr_db": t,
                     "rejected_by_gate": float(1 - ok.mean()),
                     "rejected_genuine": float(1 - ok[gen].mean()), "rejected_impostor": float(1 - ok[~gen].mean()),
                     **_summary(dist[gen & ok], dist[~gen & ok])})
    print(name, "done", flush=True)


def run_aggregation(rows: list[dict]) -> None:
    """Per speaker: utterances 0,1 enroll; the remaining ones form consecutive probe pairs (2,3), (4,5), ..."""
    d = load("voice")
    E, labels = d["embeddings"], d["labels"]
    speakers = sorted(set(labels.tolist()))
    idx = {s: np.nonzero(labels == s)[0] for s in speakers}
    enroll = {s: {1: E[idx[s][0]], 2: centroid_embedding([E[idx[s][0]], E[idx[s][1]]])} for s in speakers}
    pairs = {s: [(idx[s][k], idx[s][k + 1]) for k in range(2, len(idx[s]) - 1, 2)] for s in speakers}
    for n_enroll in (1, 2):
        for n_probe in (1, 2):
            gen, imp = [], []
            for target in speakers:
                P, is_gen = [], []
                for claimant in speakers:
                    for a, b in pairs[claimant]:
                        probe_list = [E[a], E[b]] if n_probe == 1 else [centroid_embedding([E[a], E[b]])]
                        P += probe_list
                        is_gen += [claimant == target] * len(probe_list)
                dist, is_gen = _distances(enroll[target][n_enroll], np.stack(P), target), np.array(is_gen)
                gen += dist[is_gen].tolist()
                imp += dist[~is_gen].tolist()
            rows.append({"evidence_label": REAL, "experiment": "aggregation", "condition":
                         f"enroll_{n_enroll}_utterance{'s' if n_enroll > 1 else ''}__probe_{n_probe}_utterance{'s' if n_probe > 1 else ''}",
                         **_summary(np.array(gen), np.array(imp))})
            print(rows[-1], flush=True)


def run_latency(rows: list[dict], runs: int = 30) -> None:
    from embeddings.pipelines import VoicePipeline
    from preprocessing.voice import VoicePreprocessor, load_wav_file

    d = load("voice")
    pipe, pre = VoicePipeline(), VoicePreprocessor()
    clips = [load_wav_file(p) for p in d["paths"][:2]]
    ref = d["embeddings"][0]

    def once(n_utterances: int) -> tuple[float, float, float]:
        t0 = time.perf_counter()
        for w, sr in clips[:n_utterances]:
            pre.quality_metrics(w, sr)
        t1 = time.perf_counter()
        embs = [pipe.embed(w, sample_rate=sr) for w, sr in clips[:n_utterances]]
        t2 = time.perf_counter()
        _distance(ref, centroid_embedding(embs), "latency-user")
        t3 = time.perf_counter()
        return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t0) * 1000

    once(1)  # warm-up
    for n in (1, 2):
        m = np.array([once(n) for _ in range(runs)])
        for col, stage in enumerate(("quality_metrics", "embedding_extraction", "end_to_end_voice_decision")):
            rows.append({"evidence_label": SOFTWARE, "experiment": "latency", "condition": f"{n}_utterance{'s' if n > 1 else ''}",
                         "stage": stage, "runs": runs, "mean_ms": float(m[:, col].mean()), "median_ms": float(np.median(m[:, col])),
                         "p95_ms": float(np.percentile(m[:, col], 95))})
        print(rows[-1], flush=True)


def main() -> None:
    rows: list[dict] = []
    run_aggregation(rows)
    run_gate_and_duration(rows)
    run_latency(rows)
    write_csv(RESULTS / "voice_capture_pipeline.csv", rows, REAL)


if __name__ == "__main__":
    main()
