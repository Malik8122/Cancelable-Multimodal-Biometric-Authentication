"""Calibrate a protected-template (BioHash Hamming-similarity) threshold for
voice, anchored to real operating points read off its actual ROC curve
(evaluation/results/voice_roc.csv, from real Kaggle test-set evaluation) -
not a guess, and not the EER crossover either (see below for why).

Voice only: face has no ROC curve to anchor to. The anchoring machinery is
kept generic (it takes `modality`/`embedding_dim` as parameters), but
`main()` below only invokes it for voice, and applying this threshold
change was an explicit, disclosed, user-approved decision - see
docs/AUTHENTICATION_RELIABILITY_REPORT.md for the exact tradeoff (far/frr)
accepted.

Background (see docs/AUTHENTICATION_RELIABILITY_REPORT.md for the full
writeup): the backend was falling back to Settings.match_threshold=0.9 for
every modality because no evaluation/results/<modality>_threshold.json
existed. 0.9 was never validated against the score space authentication
actually compares (protected-template Hamming similarity), and real project
data shows it's the wrong number in both directions depending on modality:
voice's real raw-embedding EER operating point is at cosine~0.32 (nowhere
near the ~1.0 that 0.9 implicitly assumes).

Why the EER crossover is the wrong anchor for a security threshold: EER is
where false-accept and false-reject rates are *equal*, not where they're
*low*. For a weak checkpoint (a high EER), anchoring "genuine" there would
let a matching share of impostor attempts clear that threshold - weakening
security, not fixing reliability. This script instead reads: (a)
the lowest threshold that still keeps FAR at or below 5% in the real ROC
curve, as the "genuine" anchor - a security-conscious operating point, not
the weakest one that happens to balance errors - and (b) the real ROC
curve's own median-impostor operating point (FAR=50%), as the "impostor"
anchor - honest about how much real overlap the checkpoint actually has,
rather than assuming a best-case separation that isn't real.

**Critical pairing detail, fixed after an earlier version of this script got
it wrong**: `backend/services/base_service.py::authenticate` always derives
the comparison key from the *claimed* user_id, for both the stored template
and the freshly-generated candidate - so a real impostration attempt (person
B presenting their own biometric while claiming to be user A) compares B's
embedding against A's template *under A's key*, not under two different
keys. An earlier version of this script built genuine/impostor pairs across
several different labels, each with its *own* derived key (mirroring
evaluation/threshold_calibration.py's general-purpose pairing) - that
measures a materially easier question (does a different *key* mask a
similar embedding?) than the one that actually matters here (does the same
*key*'s BioHash projection still separate two different people's raw
embeddings?), and produced a threshold that looked cleanly separated in
calibration but let a same-key impostor through in a real
ModalityService.authenticate() check. This version builds every genuine AND
impostor pair under one shared per-"victim" key, matching the real
threat model exactly.

Why this script exists instead of running evaluation/threshold_calibration.py
directly on real embeddings: the raw per-sample embeddings from the Kaggle
evaluation runs were never persisted (only the aggregate ROC/metrics were
saved), and regenerating them means re-downloading the datasets and
re-running inference - out of scope for this reliability sprint. A prior
attempt to substitute purely synthetic (noise/geometric-pattern) "identities"
run through the real trained networks was tried and rejected (see
tests/test_authentication_debug_sprint.py's docstring): these specific
networks collapse any out-of-distribution synthetic input toward
near-identical embeddings regardless of "identity", making a
network-generated synthetic impostor distribution meaningless.

What this script does instead: construct embedding-space vectors directly
(never touching the neural networks, sidestepping that OOD-collapse problem
entirely) whose cosine similarities are set, by exact geometric
construction, to the real genuine/impostor operating points above.
`derive_key` / `generate_template` / `compare` below are the unmodified real
production code; only the *input similarity values* are synthetic, and both
are pinned to real, previously measured facts about this project's own
trained models - not chosen to produce a target answer. This is
real-anchored calibration, not a real multi-sample-per-identity dataset -
the output JSON says so explicitly, and should be replaced the moment such a
dataset is available (no code change needed for that: backend/threshold_loader.py
already reads whatever <modality>_threshold.json exists).

Face is deliberately left untouched: evaluation/results/face_metrics.csv has
no ROC curve to anchor to (only accuracy/eer/auc), and every real test in
this project's debug sprints has shown face reliably authenticating at the
existing 0.9 fallback - there is no measured problem there to fix.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from evaluation.metrics import accuracy_at_threshold, compute_eer, confusion_matrix_at_threshold
from evaluation.roc import compute_roc
from evaluation.threshold_calibration import sweep_thresholds
from template_protection.biohash import generate_template
from template_protection.hkdf_keys import derive_key
from template_protection.matcher import compare
from template_protection.utils import l2_normalize

MASTER_SECRET = "calibration-script-secret-not-for-production"
APPLICATION_ID = "capstone-demo"
OUTPUT_BITS = 128
RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results"

NUM_VICTIMS = 40
GENUINE_SAMPLES_PER_VICTIM = 4
IMPOSTOR_SAMPLES_PER_VICTIM = 4


def _read_roc_operating_point(modality: str, *, max_far: float | None = None, target_fpr: float | None = None) -> float:
    """Read a real raw-embedding cosine-similarity operating point off the
    modality's actual ROC curve (evaluation/results/<modality>_roc.csv, from
    real Kaggle test-set evaluation).

    `max_far`: the lowest-threshold row whose FAR is still <= `max_far` - the
    boundary of that constraint (most permissive while still safe), i.e.
    "how similar do genuine pairs need to be to clear a threshold that keeps
    false accepts at or below this rate."
    `target_fpr`: the row whose FAR is closest to `target_fpr` - used to read
    off a realistic *impostor* operating point (e.g. the median of the real
    impostor distribution, at target_fpr=0.5), rather than assuming impostor
    pairs sit at cosine~0, which is false for a weak checkpoint.
    """
    path = RESULTS_DIR / f"{modality}_roc.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows = [r for r in rows if r["threshold"] not in ("inf", "-inf")]
    if max_far is not None:
        candidates = [r for r in rows if float(r["fpr"]) <= max_far]
        chosen = min(candidates, key=lambda r: float(r["threshold"])) if candidates else min(rows, key=lambda r: float(r["fpr"]))
    else:
        chosen = min(rows, key=lambda r: abs(float(r["fpr"]) - target_fpr))
    return float(chosen["threshold"])


def _vector_with_cosine(base: np.ndarray, target_cosine: float, seed: int) -> np.ndarray:
    """A unit vector at exactly `target_cosine` similarity to `base` (`base` must be unit-norm).

    Deterministic geometric construction (not a search/guess): pick a random
    direction orthogonal to `base`, then combine `base` and that orthogonal
    direction with coefficients target_cosine and sqrt(1 - target_cosine^2) -
    a linear combination of two orthonormal vectors with squared coefficients
    summing to 1 is itself unit-norm, and its dot product with `base` is
    exactly `target_cosine` by construction.
    """
    rng = np.random.default_rng(seed)
    random_vec = rng.standard_normal(base.shape[0])
    orthogonal = random_vec - np.dot(random_vec, base) * base
    orthogonal = orthogonal / np.linalg.norm(orthogonal)
    return target_cosine * base + np.sqrt(max(0.0, 1.0 - target_cosine**2)) * orthogonal


def _same_key_genuine_impostor_scores(
    modality: str, embedding_dim: int, genuine_cosine: float, impostor_cosine: float, output_bits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Real threat-model pairing: every genuine AND impostor comparison for
    one "victim" happens under that victim's own key - matching exactly what
    backend/services/base_service.py::authenticate does (it derives the
    comparison key from the *claimed* user_id, so an impostor's candidate
    template is generated under the *victim's* key, never their own).
    """
    genuine_scores, impostor_scores = [], []

    for victim in range(NUM_VICTIMS):
        victim_base = l2_normalize(np.random.default_rng(victim).standard_normal(embedding_dim))
        key = derive_key(MASTER_SECRET, application_id=APPLICATION_ID, user_id=f"victim-{victim}", modality=modality)
        stored_template = generate_template(victim_base, key, output_bits=output_bits)

        for sample in range(GENUINE_SAMPLES_PER_VICTIM):
            recapture = _vector_with_cosine(victim_base, genuine_cosine, seed=victim * 10_000 + sample)
            candidate = generate_template(recapture, key, output_bits=output_bits)
            genuine_scores.append(compare(candidate, stored_template, metric="hamming"))

        for sample in range(IMPOSTOR_SAMPLES_PER_VICTIM):
            attacker = _vector_with_cosine(victim_base, impostor_cosine, seed=victim * 10_000 + 5_000 + sample)
            candidate = generate_template(attacker, key, output_bits=output_bits)
            impostor_scores.append(compare(candidate, stored_template, metric="hamming"))

    return np.array(genuine_scores), np.array(impostor_scores)


def calibrate(modality: str, embedding_dim: int, *, genuine_max_far: float) -> dict:
    genuine_cosine = _read_roc_operating_point(modality, max_far=genuine_max_far)
    impostor_cosine = _read_roc_operating_point(modality, target_fpr=0.5)

    genuine_scores, impostor_scores = _same_key_genuine_impostor_scores(
        modality, embedding_dim, genuine_cosine, impostor_cosine, OUTPUT_BITS
    )

    sweep = sweep_thresholds(genuine_scores, impostor_scores)
    eer, eer_threshold = compute_eer(genuine_scores, impostor_scores)
    roc = compute_roc(genuine_scores, impostor_scores)
    best_row = min(sweep, key=lambda row: abs(row["far"] - row["frr"]))

    summary = {
        "modality": modality,
        "threshold": best_row["threshold"],
        "eer": eer,
        "eer_threshold": eer_threshold,
        "far": best_row["far"],
        "frr": best_row["frr"],
        "accuracy": best_row["accuracy"],
        "auc": roc["auc"],
        "num_genuine_pairs": len(genuine_scores),
        "num_impostor_pairs": len(impostor_scores),
        "output_bits": OUTPUT_BITS,
        "calibration_method": "real_roc_anchored_synthetic_same_key",
        "note": (
            f"Genuine pairs constructed at cosine similarity={genuine_cosine:.4f} (the real raw-embedding "
            f"operating point with FAR<={genuine_max_far:.0%}) and impostor pairs at cosine={impostor_cosine:.4f} "
            f"(the real median impostor operating point) - both read from evaluation/results/{modality}_roc.csv, "
            "actual Kaggle test-set evaluation, not a real multi-sample-per-identity dataset. Every genuine and "
            "impostor comparison for one synthetic 'victim' uses that victim's own derived key, matching "
            "backend/services/base_service.py::authenticate's real pairing (the candidate is always compared "
            "under the *claimed* identity's key). See scripts/calibrate_protected_thresholds.py's docstring for "
            "the full rationale. Replace with real-dataset calibration when available - no code change needed, "
            "backend/threshold_loader.py reads whatever this file contains."
        ),
    }

    (RESULTS_DIR / f"{modality}_threshold.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = RESULTS_DIR / f"{modality}_protected_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["threshold", "accuracy", "far", "frr", "eer", "tp", "fp", "tn", "fn"])
        for row in sweep:
            (tn, fp), (fn, tp) = confusion_matrix_at_threshold(genuine_scores, impostor_scores, row["threshold"])
            writer.writerow([row["threshold"], row["accuracy"], row["far"], row["frr"], eer, int(tp), int(fp), int(tn), int(fn)])

    return summary


def main() -> None:
    # Voice only: face has no ROC curve to anchor to, so it is deliberately
    # left untouched. This is an explicit, reviewed, user-approved change to
    # voice's threshold only - see docs/AUTHENTICATION_RELIABILITY_REPORT.md.
    for modality, embedding_dim in [("voice", 192)]:
        summary = calibrate(modality, embedding_dim, genuine_max_far=0.05)
        print(
            f"{modality}: threshold={summary['threshold']:.4f}  eer={summary['eer']:.4f}  "
            f"far={summary['far']:.4f}  frr={summary['frr']:.4f}  auc={summary['auc']:.4f}  "
            f"({summary['num_genuine_pairs']} genuine / {summary['num_impostor_pairs']} impostor pairs)"
        )
    print("\nWrote evaluation/results/voice_threshold.json + voice_protected_metrics.csv")
    print("Face intentionally left uncalibrated (no ROC curve to anchor to).")


if __name__ == "__main__":
    main()
