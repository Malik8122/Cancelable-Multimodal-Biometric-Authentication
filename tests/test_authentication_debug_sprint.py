"""Regression tests from the "genuine user is being denied" debug sprint.

Root-caused via `scripts/debug_authentication_pipeline.py` (kept in the repo
so this trace can be re-run against any future checkpoint/preprocessing
change): two distinct, real contributing causes were found.

1. **A concrete bug** in `preprocessing/voice.py::VoicePreprocessor._trim_silence`
   (fixed in this sprint - see `tests/test_voice_preprocessing.py`'s new
   tests): overlapping voiced frames were concatenated directly instead of
   merged into a per-sample mask, inflating a fully-voiced 4-second clip to
   ~2.5x its length. `_fixed_length_segment`'s center-crop then selected from
   that unstable, duplicated sequence, so a tiny amount of realistic capture
   noise produced a wildly different final segment - observed as a genuine
   same-speaker cosine similarity of -0.18 instead of ~1.0.

2. **A separate, pre-existing, NOT fixed here** gap: `Settings.match_threshold`
   (0.9) is applied as a fallback to every modality whenever
   `evaluation/results/<modality>_threshold.json` doesn't exist yet - which is
   currently the case for both face and voice (confirmed: the
   directory holds only the *_metrics.csv/*_roc.csv files from the Kaggle
   *raw-embedding* evaluation runs, no *_threshold.json calibrated on
   protected-template Hamming scores). 0.9 was never validated against that
   score space; `config.py`'s own comment already flags it as "a conservative
   starting point... real deployments should tune this... rather than
   trusting this default blindly." Combined with `ALL_REQUIRED` fusion policy
   (every submitted modality must individually pass), a single modality
   sitting a little under 0.9 - which the debug script shows even a *fixed*
   voice pipeline can genuinely do (~0.84-0.88 Hamming similarity for a
   realistic small amount of capture variation) - is enough to deny an
   otherwise-genuine multi-factor authentication.

   This second gap is a **data** problem (no real, labeled, multi-sample
   dataset is available in this environment to calibrate against - see
   `evaluation/threshold_calibration.py`'s own docstring on why calibrating
   against synthetic data would produce "a mathematically valid but
   biometrically meaningless number"), not a code bug, so it is deliberately
   NOT "fixed" by lowering `match_threshold` here - that would be exactly the
   "do not lower thresholds blindly" / "do not fake success" thing this sprint
   was told not to do. `get_modality_threshold` already picks up a real
   calibration file automatically the moment one is produced - no code change
   needed for that later.

A follow-up reliability sprint investigated real-data-anchored calibration
further (see `scripts/calibrate_protected_thresholds.py`, uncommitted output)
and surfaced another important, real finding, captured here as a permanent
regression test:

3. A calibration methodology pitfall: pairing genuine/impostor samples under
   *different* per-identity keys (as `evaluation/threshold_calibration.py`'s
   general-purpose pairing does) measures an easier question than what this
   system's real authenticate() flow does - it always compares under the
   *claimed* identity's one key, for both the genuine candidate and any
   impostor's candidate. A threshold calibrated against different-key pairs
   looked cleanly separated but let a same-key impostor through in a direct
   `ModalityService.authenticate()` check. `test_a_same_key_impostor_...`
   below pins the *correct* (same-key) threat model as a permanent test.
"""

from __future__ import annotations

import numpy as np
import pytest

from fusion.config import FusionPolicy
from fusion.policy import evaluate_fusion_policy

APPLICATION_ID = "capstone-demo"


def _stub_face_service():
    """The real ModalityService for face over tests/test_flexible_auth.py's deterministic stub embedder (MTCNN
    rejects synthetic images, so a real face roundtrip cannot run offline)."""
    from backend.config import get_settings
    from backend.services.base_service import ModalityService
    from tests.test_flexible_auth import _StubPipeline

    return ModalityService("face", _StubPipeline(512), get_settings())


def _noisy(image: np.ndarray, scale: float, seed: int) -> np.ndarray:
    """A synthetic stand-in for "the same physical biometric, captured again":
    small additive pixel noise, representing sensor noise / minor
    lighting-or-placement differences between two real captures."""
    rng = np.random.default_rng(seed)
    return np.clip(image.astype(np.float32) + rng.normal(scale=scale, size=image.shape), 0, 255).astype(np.uint8)


def test_genuine_user_authenticates_with_a_second_slightly_varied_face_capture(db_session):
    """The literal bug report: enroll once, then authenticate with a capture
    that is similar but *not byte-identical* (a real second capture would never
    be pixel-perfect) - this must still succeed. Bypasses MTCNN detection/alignment
    (which correctly rejects any non-face synthetic image, see
    tests/test_preprocessing.py) by calling FaceEmbedder directly on an
    already-"aligned"-shaped array, so this exercises the real trained face
    network + real BioHash pipeline without needing a real photograph."""
    pytest.importorskip("facenet_pytorch", reason="facenet-pytorch not installed in this environment")
    from backend.config import get_settings
    from backend.services.base_service import ModalityService
    from models.face.inference import FaceEmbedder

    settings = get_settings()
    embedder = FaceEmbedder(checkpoint_path=settings.face_model_path)
    assert embedder.mock_mode is False, "face checkpoint failed to load - falling back to mock mode"

    class _DirectEmbedPipeline:
        def embed(self, image: np.ndarray) -> np.ndarray:
            return embedder.extract_embedding(image)

    service = ModalityService("face", _DirectEmbedPipeline(), settings)

    rng = np.random.default_rng(2)
    enrolled_face = rng.integers(0, 255, (160, 160, 3), dtype=np.uint8)
    service.enroll(db_session, enrolled_face, user_id="U-GENUINE-FACE", application_id=APPLICATION_ID)

    recaptured = _noisy(enrolled_face, scale=3.0, seed=3)
    result = service.authenticate(db_session, recaptured, user_id="U-GENUINE-FACE", application_id=APPLICATION_ID)

    assert result.authenticated is True
    assert result.score >= result.threshold


def test_a_never_enrolled_user_cannot_authenticate(db_session):
    """"Different user is rejected", in the sense this system actually
    implements: authentication is *verification against a claimed identity*
    (there is no "identify who this biometric belongs to among all users"
    mode - see backend/services/base_service.py's docstring), so the
    unambiguous, always-true version of "not the enrolled user" is: no
    active template exists for the claimed user_id at all."""
    image = np.random.default_rng(1).integers(0, 256, size=(300, 300, 3), dtype=np.uint8)
    service = _stub_face_service()
    service.enroll(db_session, image, user_id="U-REAL", application_id=APPLICATION_ID)

    result = service.authenticate(db_session, image, user_id="someone-else-entirely", application_id=APPLICATION_ID)

    assert result.authenticated is False
    assert result.score == 0.0


def test_all_required_denies_the_exact_mixed_result_this_bug_report_describes():
    """Direct regression test for the bug report's own scenario: a genuine
    user's face individually passes, but voice - even
    with the `_trim_silence` fix - can genuinely sit a little under the
    uncalibrated 0.9 fallback threshold (measured ~0.84-0.88 for a realistic
    small amount of capture variation, via
    scripts/debug_authentication_pipeline.py). Under ALL_REQUIRED (the
    secure default), that must still deny the whole multi-factor attempt -
    this is the fusion *policy* working exactly as designed; the actual fix
    needed for this specific numeric gap is real threshold calibration data,
    not a fusion-policy change (see this module's docstring)."""
    scores = {"face": 1.0, "voice": 0.86}
    individually_authenticated = {"face": True, "voice": False}

    decision = evaluate_fusion_policy(scores, individually_authenticated, FusionPolicy.ALL_REQUIRED, fusion_threshold=0.9)

    assert decision.authenticated is False
    assert decision.failed_modalities == ["voice"]
    assert decision.matched_modalities == ["face"]
    # The averaged fused_score (0.93) would itself have cleared 0.9 - proof
    # this is ALL_REQUIRED correctly refusing to let a strong modality
    # compensate for the one that individually failed, not a scoring error.
    assert decision.fused_score > 0.9


def test_all_required_denies_a_real_user_reported_incident_with_a_one_bit_margin():
    """Regression test for a real, live-diagnosed incident (see
    docs/AUTHENTICATION_RELIABILITY_REPORT.md's "Incident" section): a
    genuinely enrolled user's UI showed Face=0.898/threshold=0.900/No Match
    and Voice=0.719/threshold=0.900/No Match, Fusion=Denied, for a person
    they stated was the same one enrolled.

    Full re-investigation (preprocessing determinism, key derivation,
    template retrieval against the actual real database row - user
    DEMO-YWKQMU7B, template_version=1/key_version=3, matching the UI exactly -
    fusion decision logic) found every mechanical component correct. The
    diagnostic signature that rules out a code bug: 0.898 is a *rounded
    display* of a Hamming score that, on a 128-bit template, can only be an
    exact multiple of 1/128 - the true value was 115/128 (0.8984375),
    exactly ONE bit-flip short of the 116/128 (0.90625) needed to clear 0.9.
    A real preprocessing/template/model bug in this codebase has a very
    different signature: the two bugs actually found and fixed in earlier
    sprints (preprocessing/voice.py's frame-overlap and mel-padding issues)
    each collapsed genuine similarity to near-chance or *negative* cosine
    similarity - nothing like a single-bit miss. Voice's 92/128 (0.71875,
    36 bits differing) is a larger gap, consistent with voice's
    already-documented, already-disclosed uncalibrated-threshold gap (see
    scripts/calibrate_protected_thresholds.py's investigation), not a new
    regression.

    This test pins the correct, current, intentional behavior given the
    real reported scores and the unmodified 0.9 threshold - it must keep
    passing (denied) unless a real, disclosed threshold-calibration change
    is deliberately made, and must never be "fixed" by editing the scores
    here to force it to pass.
    """
    scores = {"face": 115 / 128, "voice": 92 / 128}
    individually_authenticated = {"face": False, "voice": False}

    decision = evaluate_fusion_policy(scores, individually_authenticated, FusionPolicy.ALL_REQUIRED, fusion_threshold=0.9)

    assert decision.authenticated is False
    assert decision.failed_modalities == ["face", "voice"]
    assert decision.matched_modalities == []
    # Precisely pins the one-bit-margin finding: 115/128 correctly fails,
    # but a single additional matching bit (116/128) would have passed -
    # face was this close to passing on its own, not a wide miss.
    from template_protection.matcher import accept

    assert accept(115 / 128, threshold=0.9) is False
    assert accept(116 / 128, threshold=0.9) is True


def _unit_vector(dim: int, seed: int) -> np.ndarray:
    from template_protection.utils import l2_normalize

    return l2_normalize(np.random.default_rng(seed).standard_normal(dim))


def _vector_with_cosine(base: np.ndarray, target_cosine: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_vec = rng.standard_normal(base.shape[0])
    orthogonal = random_vec - np.dot(random_vec, base) * base
    orthogonal = orthogonal / np.linalg.norm(orthogonal)
    return target_cosine * base + np.sqrt(max(0.0, 1.0 - target_cosine**2)) * orthogonal


def test_a_same_key_impostor_with_realistic_similarity_is_rejected_at_the_current_threshold(db_session):
    """The real threat this system defends against: `authenticate()` derives
    the comparison key from the *claimed* user_id (see
    backend/services/base_service.py), so an impostor presenting their own
    biometric while claiming to be the victim is compared under the VICTIM'S
    key, not their own - a materially different (harder) test than "does a
    different key mask a similar embedding", which
    evaluation/threshold_calibration.py's general-purpose pairing measures
    instead (a distinction this sprint's investigation found the hard way -
    see scripts/calibrate_protected_thresholds.py's docstring; an earlier,
    incorrect calibration attempt would have let an impostor constructed this
    exact way authenticate).

    Face decides on the calibrated estimate of the embedding cosine
    (`Settings.face_cosine_threshold`); the impostor's raw-embedding cosine
    similarity to the victim is 0.2 below that threshold - far above chance
    (a strongly similar embedding), not a best-case near-zero assumption.
    Compared under the victim's key, it is still correctly denied - proving
    the face decision rule has not been silently weakened.
    """
    from backend.config import get_settings
    from backend.services.base_service import ModalityService

    settings = get_settings()

    class _DirectEmbedPipeline:
        def __init__(self, vector: np.ndarray):
            self.vector = vector

        def embed(self, _raw_image: np.ndarray) -> np.ndarray:
            return self.vector

    victim_base = _unit_vector(512, seed=1)
    pipeline = _DirectEmbedPipeline(victim_base)
    service = ModalityService("face", pipeline, settings)
    placeholder_raw = np.zeros((1, 1, 3), dtype=np.uint8)

    service.enroll(db_session, placeholder_raw, user_id="VICTIM", application_id=APPLICATION_ID)

    # A strongly similar impostor - not a best-case near-zero assumption.
    pipeline.vector = _vector_with_cosine(victim_base, target_cosine=settings.face_cosine_threshold - 0.2, seed=2)
    result = service.authenticate(db_session, placeholder_raw, user_id="VICTIM", application_id=APPLICATION_ID)

    assert result.authenticated is False


def test_a_same_key_voice_impostor_with_realistic_similarity_is_rejected():
    """Same real threat-model check as
    `test_a_same_key_impostor_with_realistic_similarity_is_rejected_at_the_current_threshold`
    (face), for voice: confirms the preprocessing fix in
    preprocessing/voice.py::VoicePreprocessor.preprocess (which only changes
    how a genuine embedding is *computed* from real audio) does not touch
    template_protection's matching behavior at all. voice's real, measured
    median impostor cosine similarity is ~0.00 (evaluation/results/voice_roc.csv) -
    used directly here rather than a full synthetic-audio round trip, because
    synthetic broadband "speaker" patterns carry no real speaker-identity
    signal for this network to discriminate (confirmed separately: two
    differently-seeded synthetic waveforms both incorrectly authenticated
    against each other's real trained-model embeddings - an orthogonal,
    already-known limitation of testing without real voice data, not a
    security regression)."""
    from template_protection.hkdf_keys import derive_key
    from template_protection.biohash import generate_template
    from template_protection.matcher import compare, accept
    from template_protection.utils import l2_normalize

    victim = l2_normalize(np.random.default_rng(1).standard_normal(192))
    key = derive_key(
        "unit-test-master-secret-not-for-production", application_id=APPLICATION_ID, user_id="VICTIM", modality="voice"
    )
    stored = generate_template(victim, key, output_bits=128)

    attacker = _vector_with_cosine(victim, target_cosine=0.0, seed=2)
    score = compare(generate_template(attacker, key, output_bits=128), stored, metric="hamming")

    assert accept(score, threshold=0.9) is False


def test_fusion_only_requires_the_modalities_actually_submitted():
    """Scenario D from the reliability sprint: a face-only building must not
    require voice just because ALL_REQUIRED is the default policy.
    `evaluate_fusion_policy`'s docstring already states this ("a modality that
    was never provided must never appear here"), and backend/api/fusion.py
    only populates `scores`/`individually_authenticated` from whichever
    images/audio were actually uploaded - this test pins that contract at the
    policy layer directly, independent of the HTTP route."""
    scores = {"face": 0.97}
    individually_authenticated = {"face": True}

    decision = evaluate_fusion_policy(scores, individually_authenticated, FusionPolicy.ALL_REQUIRED, fusion_threshold=0.9)

    assert decision.authenticated is True
    assert decision.matched_modalities == ["face"]
    assert "voice" not in decision.matched_modalities and "voice" not in decision.failed_modalities
