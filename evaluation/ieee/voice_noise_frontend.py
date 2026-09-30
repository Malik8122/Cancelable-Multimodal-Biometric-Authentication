"""Voice noise robustness of candidate signal front-ends (EXPERIMENTAL; does not change the production pipeline).

Question: can a light, deterministic clean-up step applied to EVERY recording (enrollment and authentication alike)
make genuine voice attempts pass more often in everyday noise, without raising false accepts?

Front-ends (applied to the raw waveform before the unchanged VoicePipeline.embed; the reference enrollment clip gets
the same front-end as the probes, as it would in deployment after re-enrollment):
  none       production (no clean-up)
  hp80       4th-order Butterworth high-pass at 80 Hz, zero-phase: removes hum / fan / handling rumble below the
             voice fundamental (adult speech F0 is >= ~85 Hz)
  hp80_ss    hp80 + mild spectral subtraction: per-frequency noise floor = 10th percentile of the frame power over the
             recording, Wiener-style gain max(1 - N/P, 0.1) (at most -20 dB attenuation), same STFT framing both ways
  metricgan  neural speech enhancement: SpeechBrain MetricGAN+ (speechbrain/metricgan-plus-voicebank, trained on
             VoiceBank-DEMAND at 16 kHz) - a learned spectral mask that keeps speech and suppresses other sound

Noise conditions (added to the PROBE only; the enrollment clip stays clean, as when a user enrolls in a quiet room and
authenticates somewhere noisier):
  clean, white_20dB, white_10dB, hum_10dB (50 Hz mains + harmonics + low-passed rumble), pink_15dB (broadband room
  noise), babble_15dB (three other speakers' clips mixed)

Data and protocol exactly as evaluation/ieee/voice_capture_pipeline.py: VoxCeleb1 Indian-celebrity TEST split
(24 speakers, closed-set), reference = first clip of each speaker, 200 genuine + 200 impostor probes, protected domain
(256-bit BioHash under the target's key, calibrated estimated distance), operating rule `distance <= 0.75`. The
capture-quality gate verdict (estimated SNR >= 13 dB, measured on the front-end's output) is reported alongside.

Output: evaluation/results/voice_noise_frontend.csv (evidence label REAL DATA). Embeddings are cached in
data/eval_cache/voice_noise_frontend.npz so an interrupted run resumes.

Run: python -m evaluation.ieee.voice_noise_frontend
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, istft, sosfiltfilt, stft

from evaluation.ieee.common import CACHE, REAL, RESULTS, SEED, eer, rates, write_csv
from evaluation.ieee.experiments import load
from evaluation.ieee.robustness import _mix, _noise
from evaluation.ieee.voice_capture_pipeline import _distances, _probe_subset

THRESHOLD = 0.75
GATE_DB = 13.0
FRONTENDS = ("none", "hp80", "hp80_ss", "metricgan")
CONDITIONS = ("clean", "white_20dB", "white_10dB", "hum_10dB", "pink_15dB", "babble_15dB")
CACHE_FILE = CACHE / "voice_noise_frontend.npz"


def highpass(w: np.ndarray, sr: int, cutoff: float = 80.0) -> np.ndarray:
    sos = butter(4, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, w).astype(np.float32)


def spectral_subtract(w: np.ndarray, sr: int, floor: float = 0.1) -> np.ndarray:
    nper = int(0.032 * sr)
    _, _, Z = stft(w, fs=sr, nperseg=nper, noverlap=nper * 3 // 4)
    power = np.abs(Z) ** 2
    noise = np.percentile(power, 10, axis=1, keepdims=True)
    gain = np.maximum(1.0 - noise / np.maximum(power, 1e-20), floor)
    _, y = istft(Z * gain, fs=sr, nperseg=nper, noverlap=nper * 3 // 4)
    return np.asarray(y[: len(w)], np.float32)


_ENHANCER = None


def metricgan(w: np.ndarray, sr: int) -> np.ndarray:
    global _ENHANCER
    import torch

    if _ENHANCER is None:
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
        from speechbrain.utils.fetching import LocalStrategy

        _ENHANCER = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank", savedir=str(CACHE / "sb_metricgan"), local_strategy=LocalStrategy.COPY)
    assert sr == 16000, "MetricGAN+ expects 16 kHz"
    with torch.no_grad():
        y = _ENHANCER.enhance_batch(torch.as_tensor(np.asarray(w, np.float32))[None, :], lengths=torch.tensor([1.0]))
    return y[0].numpy().astype(np.float32)


def apply_frontend(name: str, w: np.ndarray, sr: int) -> np.ndarray:
    if name == "none":
        return np.asarray(w, np.float32)
    if name == "metricgan":
        return metricgan(w, sr)
    x = highpass(np.asarray(w, np.float64), sr)
    return spectral_subtract(x, sr) if name == "hp80_ss" else x


def _scaled(w: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    ps, pn = np.mean(w**2) + 1e-12, np.mean(noise**2) + 1e-12
    return w + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def add_noise(condition: str, w: np.ndarray, sr: int, rng, others: list[np.ndarray]) -> np.ndarray:
    w = np.asarray(w, np.float64)
    n = len(w)
    if condition == "clean":
        return w
    if condition.startswith("white"):
        return _noise(w, float(condition.split("_")[1][:-2]), rng)
    if condition == "hum_10dB":
        t = np.arange(n) / sr
        hum = sum(a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f, a in ((50, 1.0), (100, 0.5), (150, 0.3)))
        rumble = sosfiltfilt(butter(2, 120, btype="lowpass", fs=sr, output="sos"), rng.standard_normal(n))
        rumble *= np.std(hum) / (np.std(rumble) + 1e-12)
        return _scaled(w, hum + rumble, 10.0)
    if condition == "pink_15dB":
        spectrum = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / sr)
        spectrum[1:] /= np.sqrt(f[1:])
        spectrum[0] = 0
        return _scaled(w, np.fft.irfft(spectrum, n), 15.0)
    if condition == "babble_15dB":
        picks = rng.choice(len(others), 3, replace=False)
        babble = sum(np.resize(others[k], n) / (np.std(others[k]) + 1e-12) for k in picks)
        return _mix(w, babble, 15.0)
    raise KeyError(condition)


def _summary(dist: np.ndarray, gen: np.ndarray, gate_ok: np.ndarray) -> dict:
    g, i = dist[gen], dist[~gen]
    r = rates(g, i, THRESHOLD, False)
    gg = dist[gen & gate_ok]
    return {
        "n_genuine": int(gen.sum()), "n_impostor": int((~gen).sum()),
        "genuine_mean_distance": float(g.mean()), "genuine_median_distance": float(np.median(g)),
        "FRR_at_0.75": r["FRR"], "FAR_at_0.75": r["FAR"], "EER": eer(g, i, False)[0],
        "gate_pass_rate_genuine": float(gate_ok[gen].mean()),
        "FRR_at_0.75_gate_passed": float(np.mean(gg > THRESHOLD)) if len(gg) else float("nan"),
    }


def main() -> None:
    from embeddings.pipelines import VoicePipeline
    from preprocessing.voice import VoicePreprocessor, load_wav_file

    d = load("voice")
    labels, paths = d["labels"], d["paths"]
    rng = np.random.default_rng(SEED)
    refs, probes = _probe_subset(labels, rng)
    pipe, pre = VoicePipeline(), VoicePreprocessor()
    assert not pipe.is_mock, "voice checkpoint missing"
    gen = np.array([g for _, _, g in probes])

    probe_idx = {i for i, _, _ in probes} | set(refs.values())
    other_pool = [k for k in range(len(labels)) if k not in probe_idx][:60]
    others = [np.asarray(load_wav_file(paths[k])[0], np.float64) for k in other_pool]

    stored = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    rows = []
    for fe in FRONTENDS:
        ref_key = f"{fe}__refs"
        if ref_key not in stored:
            ref_embs = []
            for s in sorted(refs):
                w, sr = load_wav_file(paths[refs[s]])
                ref_embs.append(pipe.embed(apply_frontend(fe, w, sr), sample_rate=sr))
            stored[ref_key] = np.stack(ref_embs)
            np.savez(CACHE_FILE, **stored)
        E_ref = {s: stored[ref_key][k] for k, s in enumerate(sorted(refs))}

        for cond in CONDITIONS:
            key = f"{fe}__{cond}"
            if f"{key}__emb" not in stored:
                crng = np.random.default_rng([SEED, CONDITIONS.index(cond)])  # same noise for every front-end
                embs, snrs = [], []
                for idx, _, _ in probes:
                    w, sr = load_wav_file(paths[idx])
                    x = apply_frontend(fe, add_noise(cond, w, sr, crng, others).astype(np.float32), sr)
                    snrs.append(pre.quality_metrics(x, sr)["estimated_snr_db"])
                    embs.append(pipe.embed(x, sample_rate=sr))
                stored[f"{key}__emb"], stored[f"{key}__snr"] = np.stack(embs), np.array(snrs)
                np.savez(CACHE_FILE, **stored)
            embs, snrs = stored[f"{key}__emb"], stored[f"{key}__snr"]
            dist = np.empty(len(probes))
            for target in sorted(refs):
                rows_t = [k for k, (_, t, _) in enumerate(probes) if t == target]
                dist[rows_t] = _distances(E_ref[target], embs[rows_t], target)
            row = {"evidence_label": REAL, "frontend": fe, "condition": cond, **_summary(dist, gen, snrs >= GATE_DB)}
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    write_csv(RESULTS / "voice_noise_frontend.csv", rows, REAL)


if __name__ == "__main__":
    main()
