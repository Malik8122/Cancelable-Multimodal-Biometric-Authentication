"""Part 13: robustness of the protected-template decision under controlled degradations of REAL probe samples.

A fixed subset of genuine and impostor comparisons (same references, same keys as Part 4) is re-run with the probe
sample degraded before the shipped pipeline embeds it; references stay clean (they represent enrollment).
Degradations are simulated signal transforms applied to real data. Head pose and a physically different microphone
cannot be simulated faithfully and are reported as NOT AVAILABLE.
Run: python -m evaluation.ieee.robustness [face|voice ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from evaluation.ieee.common import CACHE, COLORS, DBL_W, FUTURE, REAL, RESULTS, SEED, eer, ieee_style, rates, save_fig, timed, write_csv
from evaluation.ieee.experiments import SYSTEM_RULE, comparisons, load
from evaluation.ieee.protected import BITS, estimated_cosine, estimated_distance, hamming_similarity, key_for, templates

N_PER_CLASS = 200


# ----------------------------------------------------------------------------- degradations


def _gamma(img, g):
    return np.clip(255.0 * (img / 255.0) ** g, 0, 255).astype(np.uint8)


def _rotate(img, deg, border=0):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(border,) * 3)


def _scale(img, f):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _occlude(img, top, bottom):
    out = img.copy()
    h = img.shape[0]
    out[int(h * top): int(h * bottom), :] = 0
    return out


FACE_CONDITIONS = {
    "clean": lambda x: x,
    "dark_gamma2.2": lambda x: _gamma(x, 2.2),
    "bright_gamma0.45": lambda x: _gamma(x, 0.45),
    "low_contrast_x0.4": lambda x: np.clip((x.astype(float) - x.mean()) * 0.4 + x.mean(), 0, 255).astype(np.uint8),
    "blur_sigma2": lambda x: cv2.GaussianBlur(x, (0, 0), 2),
    "blur_sigma4": lambda x: cv2.GaussianBlur(x, (0, 0), 4),
    "rotation_15deg": lambda x: _rotate(x, 15),
    "rotation_30deg": lambda x: _rotate(x, 30),
    "distance_scale0.5": lambda x: _scale(x, 0.5),
    "distance_scale0.25": lambda x: _scale(x, 0.25),
    "occlusion_eyes_band": lambda x: _occlude(x, 0.33, 0.50),
    "occlusion_lower_third": lambda x: _occlude(x, 0.62, 1.0),
}


def _noise(w, snr_db, rng):
    p = np.mean(w**2) + 1e-12
    return w + rng.standard_normal(len(w)) * np.sqrt(p / 10 ** (snr_db / 10))


def _mix(w, other, snr_db):
    other = np.resize(other, len(w))
    ps, pn = np.mean(w**2) + 1e-12, np.mean(other**2) + 1e-12
    return w + other * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def _speed(w, sr, factor):
    import torch
    import torchaudio.functional as F

    return F.resample(torch.from_numpy(w.astype(np.float32)), int(sr * factor), sr).numpy()


def _pitch(w, sr, direction):
    """Pitch shift by a factor 9/8 (+2.04 semitones) or 8/9 (-2.04) at constant duration.

    Rational resampling (scipy resample_poly) followed by a phase-vocoder time-stretch back to the original length.
    torchaudio.functional.pitch_shift is not used: for +2 semitones at 16 kHz it resamples 14254 -> 16000 Hz, whose
    tiny GCD builds a >1 GB sinc kernel per call.
    """
    import torch
    import torchaudio.functional as F
    from scipy.signal import resample_poly

    up, down = (8, 9) if direction > 0 else (9, 8)  # fewer samples played at sr = higher pitch
    y = torch.from_numpy(resample_poly(np.asarray(w, float), up, down).astype(np.float32))
    n_fft, hop = 512, 128
    window = torch.hann_window(n_fft)
    spec = torch.stft(y, n_fft, hop_length=hop, window=window, return_complex=True)
    rate = len(y) / len(w)  # < 1 lengthens, > 1 shortens
    advance = torch.linspace(0, np.pi * hop, spec.shape[-2])[..., None]
    stretched = F.phase_vocoder(spec, rate, advance)
    return torch.istft(stretched, n_fft, hop_length=hop, window=window, length=len(w)).numpy()


def _telephone(w, sr):
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, w)


def voice_conditions(rng, babble):
    return {
        "clean": lambda w, sr: w,
        "white_noise_snr20": lambda w, sr: _noise(w, 20, rng),
        "white_noise_snr10": lambda w, sr: _noise(w, 10, rng),
        "white_noise_snr5": lambda w, sr: _noise(w, 5, rng),
        "background_speech_snr10": lambda w, sr: _mix(w, babble, 10),
        "telephone_band_300_3400Hz": _telephone,
        "speed_0.9": lambda w, sr: _speed(w, sr, 0.9),
        "speed_1.1": lambda w, sr: _speed(w, sr, 1.1),
        "pitch_minus2.04_semitones": lambda w, sr: _pitch(w, sr, -1),
        "pitch_plus2.04_semitones": lambda w, sr: _pitch(w, sr, +1),
    }


# ----------------------------------------------------------------------------- runner


def _subset(c, rng):
    g = np.nonzero(c.genuine)[0]
    i = np.nonzero(~c.genuine)[0]
    return np.r_[rng.choice(g, min(N_PER_CLASS, len(g)), replace=False), rng.choice(i, min(N_PER_CLASS, len(i)), replace=False)]


def _evaluate(modality, c, sel, E, probe_embs, condition):
    name, hib, thr = SYSTEM_RULE[modality]
    ok = np.array([e is not None for e in probe_embs])
    h = np.full(len(sel), np.nan)
    for j, k in enumerate(sel):
        if not ok[j]:
            continue
        key = key_for(c.target[k], modality)
        T = templates(np.vstack([E[c.ref_index[k]], probe_embs[j]]), key)
        h[j] = hamming_similarity(T[1], T[0])
    gen = c.genuine[sel]
    s = {"hamming_similarity": h, "estimated_cosine": estimated_cosine(np.nan_to_num(h), modality),
         "estimated_distance": estimated_distance(np.nan_to_num(h), modality)}[name]
    # a failure to acquire is a rejection: count it as a non-match
    s = np.where(ok, s, -np.inf if hib else np.inf)
    r = rates(s[gen], s[~gen], thr, hib)
    valid = ok
    e = eer(s[gen & valid], s[~gen & valid], hib)[0] if (gen & valid).any() and (~gen & valid).any() else float("nan")
    return {"modality": modality, "condition": condition, "n_genuine": int(gen.sum()), "n_impostor": int((~gen).sum()),
            "failure_to_acquire_rate": float(1 - ok.mean()), "mean_genuine_hamming": float(np.nanmean(h[gen])),
            f"mean_genuine_{name}": float(np.mean(s[gen & valid])), "FRR_at_operating": r["FRR"], "FAR_at_operating": r["FAR"],
            "EER": e, "operating_rule": f"{name} {'>=' if hib else '<='} {thr}"}


def run_face():
    from embeddings.pipelines import FacePipeline

    rng = np.random.default_rng(SEED)
    c, d = comparisons("face"), load("face")
    sel = _subset(c, rng)
    pipe = FacePipeline()
    rows = []
    for name, fn in FACE_CONDITIONS.items():
        embs = []
        for k in sel:
            img = cv2.cvtColor(cv2.imread(str(d["paths"][c.probe_index[k]])), cv2.COLOR_BGR2RGB)
            try:
                embs.append(pipe.embed(fn(img)))
            except Exception:  # noqa: BLE001 - no face detected after degradation
                embs.append(None)
        rows.append(_evaluate("face", c, sel, d["embeddings"], embs, name))
        print(rows[-1], flush=True)
    rows.append({"modality": "face", "condition": "head_pose", "note": "NOT AVAILABLE - pose cannot be simulated faithfully from 2-D images", "evidence_label": FUTURE})
    return rows


def run_voice():
    from embeddings.pipelines import VoicePipeline
    from preprocessing.voice import load_wav_file

    rng = np.random.default_rng(SEED)
    c, d = comparisons("voice"), load("voice")
    sel = _subset(c, rng)
    pipe = VoicePipeline()
    # background speech: an utterance of a speaker who is neither the target nor the claimant
    rows = []
    for name in voice_conditions(rng, None):
        embs = []
        for k in sel:
            w, sr = load_wav_file(d["paths"][c.probe_index[k]])
            others = np.nonzero((d["labels"] != c.target[k]) & (d["labels"] != c.probe_identity[k]))[0]
            babble, _ = load_wav_file(d["paths"][others[k % len(others)]])
            fn = voice_conditions(rng, babble)[name]
            try:
                embs.append(pipe.embed(np.asarray(fn(np.asarray(w, float), sr), np.float32), sample_rate=sr))
            except Exception:  # noqa: BLE001
                embs.append(None)
        rows.append(_evaluate("voice", c, sel, d["embeddings"], embs, name))
        print(rows[-1], flush=True)
    rows.append({"modality": "voice", "condition": "different_physical_microphone",
                 "note": "NOT AVAILABLE - needs real re-recordings; telephone band-pass is a channel simulation only", "evidence_label": FUTURE})
    return rows


RUNNERS = {"face": run_face, "voice": run_voice}


def plot(rows):
    plt = ieee_style()
    fig, axes = plt.subplots(2, 1, figsize=(3.9, 4.6), sharex=True, gridspec_kw={"height_ratios": [12, 10]})
    for ax, m in zip(axes, ("face", "voice")):
        rr = [r for r in rows if r.get("modality") == m and r.get("FRR_at_operating") not in (None, "")]
        if not rr:
            continue
        y = np.arange(len(rr))
        ax.barh(y, [r["FRR_at_operating"] for r in rr], color=COLORS[m], height=0.6, label="FRR")
        ax.plot([r["FAR_at_operating"] for r in rr], y, "k|", ms=6, label="FAR")
        ax.set_yticks(y, [r["condition"] for r in rr], fontsize=5.5)
        ax.invert_yaxis()
        ax.set(xlim=(0, 1), title=m)
        ax.grid(True, axis="x")
    axes[-1].set_xlabel("rate at the deployed threshold")
    axes[0].legend(fontsize=6, loc="lower right")
    save_fig(fig, "fig27_robustness_frr_far", REAL, "FRR (bars) and FAR (ticks) of the deployed protected-template rule under degradations of real probes.")


if __name__ == "__main__":
    wanted = [a for a in sys.argv[1:] if a != "--plot-only"] or ([] if "--plot-only" in sys.argv else list(RUNNERS))
    out = RESULTS / "robustness.csv"
    for m in wanted:
        with timed(f"part13_robustness_{m}"):
            part = RUNNERS[m]()
        existing = []
        if out.exists():
            import csv

            with out.open(encoding="utf-8") as h:
                existing = [r for r in csv.DictReader(h) if r["modality"] != m]
            for r in existing:
                for k, v in list(r.items()):
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        pass
        write_csv(out, existing + part, REAL)
    import csv

    with out.open(encoding="utf-8") as h:
        rows = [{k: (float(v) if k in ("FRR_at_operating", "FAR_at_operating") and v not in ("", None) else v) for k, v in r.items()} for r in csv.DictReader(h)]
    plot(rows)
