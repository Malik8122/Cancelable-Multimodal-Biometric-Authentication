"""Backend configuration, loaded from environment variables / a `.env` file.

No path or secret in this module is hardcoded as a literal default that would
actually work in production: `master_secret` has **no default at all** (a
missing `MASTER_SECRET` fails fast at startup rather than silently running
with a guessable key), and every other setting's default is either a
clearly-local development value (`sqlite:///./biometric.db`) or reuses a
value already defined elsewhere in the repo (`embeddings.constants.DEFAULT_CHECKPOINTS`)
rather than a second, possibly-drifting copy of it.

See `.env.example` for the variables this reads and
`docs/BACKEND_API.md`/README for how tests supply `MASTER_SECRET` without a
real `.env` file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from embeddings.constants import DEFAULT_CHECKPOINTS


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    #: Server-side secret all HKDF key derivation is rooted in
    #: (see template_protection/hkdf_keys.py::derive_key). Required - there
    #: is no safe default for a value whose entire purpose is to be secret.
    master_secret: str

    #: SQLAlchemy database URL. SQLite by default, matching the spec's scope
    #: ("SQLite via SQLAlchemy") - swapping this to Postgres/MySQL later
    #: needs no code change beyond this URL and the SQLite-specific
    #: `connect_args` in backend/database/session.py.
    database_url: str = "sqlite:///./biometric.db"

    #: Bare filesystem path to the SQLite database file (e.g. Render's
    #: persistent disk mount, `/var/data/biometric.db`) - a deployment
    #: platform's environment-variable UI generally makes a plain path much
    #: easier to set correctly than a full SQLAlchemy URL. When set, this
    #: takes priority over `database_url` (see `resolved_database_url`);
    #: `database_url` stays the default/dev-oriented setting and is
    #: unaffected when this is left unset.
    database_path: str | None = None

    #: "development" (default) or "production" - read from `ENV`, not
    #: `ENVIRONMENT` (pydantic-settings' default case-insensitive field-name
    #: match), since that's the variable name Render deployments set.
    #: `backend/main.py` uses this only to disable interactive Swagger/ReDoc
    #: docs in production - never to change any authentication or model
    #: behavior.
    environment: str = Field(default="development", validation_alias="ENV")

    #: Default application ID used when a request doesn't specify one -
    #: primarily for local/demo use; real multi-tenant use should always pass
    #: an explicit application_id per request.
    application_id: str = "capstone-demo"

    #: Per-modality checkpoint paths, defaulting to the same
    #: `embeddings/constants.py::DEFAULT_CHECKPOINTS` locations the Colab/Kaggle
    #: training pipelines already write to - keeps one source of truth for
    #: "where does modality X's checkpoint live" rather than a second,
    #: independently-drifting copy of that mapping.
    face_model_path: Path = DEFAULT_CHECKPOINTS["face"]
    #: Face preprocessing: "bbox" (MTCNN bounding-box crop - what the deployed checkpoint was trained on) or
    #: "similarity" (5-landmark similarity alignment, preprocessing/face.py::FacePreprocessorAligned). The mode must
    #: match the preprocessing the checkpoint at face_model_path was trained with (see
    #: evaluation/reports/FACE_ALIGNMENT_DECISION.md). Env: FACE_ALIGNMENT. The mode names FACE_BASELINE (= "bbox")
    #: and FACE_ALIGNED (= "similarity") are accepted as aliases.
    face_alignment: Literal["bbox", "similarity"] = "bbox"
    voice_model_path: Path = DEFAULT_CHECKPOINTS["voice"]

    #: Default protected-template length in bits for *new* enrollments/
    #: revocations only - `ModalityService.authenticate` always regenerates
    #: the candidate template at the *stored* template's own `output_bits`
    #: (backend/services/base_service.py), never this setting, so changing
    #: this value can never cause an old template to be silently compared at
    #: the wrong bit length; it only changes what future enrollments get.
    #:
    #: Was 128 (template_protection/biohash.py::DEFAULT_OUTPUT_BITS's
    #: generic default). Raised to 256 after a real, quantified separability
    #: experiment (see docs/AUTHENTICATION_RELIABILITY_REPORT.md's "128 vs
    #: 256-bit BioHash migration" section): 128 bits measurably discarded
    #: discriminative information for both face and voice (~7 percentage
    #: points of EER each), and 256 bits recovered most of it in the same
    #: experiment. 256 stays <= face's 512-dim embedding (a single
    #: orthonormal projection block - see
    #: template_protection/transform.py::build_orthonormal_projection) but
    #: exceeds voice's 192-dim embedding (falls back to a second,
    #: non-orthogonal-to-the-first block) - an explicit, documented
    #: tradeoff, not a bug; re-measured per-modality rather than assumed.
    template_bits: int = 256

    #: Multi-template architecture (docs/MULTI_TEMPLATE_ARCHITECTURE.md):
    #: how many independent cancelable templates one enrollment capture
    #: produces per modality (T1..TN, each under its own HKDF key_version).
    #: T1 starts ACTIVE, the rest STANDBY. Env: TEMPLATE_POOL_SIZE.
    template_pool_size: int = Field(default=4, ge=1)

    #: When False (the default, and what production must use) no public
    #: response carries per-modality similarity/distance/threshold values -
    #: only the single fused similarity leaves the backend. The values are
    #: always still computed and written to the audit log. Env: DEBUG_SCORES.
    debug_scores: bool = False

    #: JSON file of building security policies (backend/buildings.py). Unset =
    #: the repository's `config/buildings.json`. Env: BUILDINGS_CONFIG_PATH.
    buildings_config_path: str | None = None

    #: Upload validation (spec: "reject oversized uploads", "validate
    #: uploaded file types").
    max_upload_size_bytes: int = 5_000_000
    allowed_content_types: tuple[str, ...] = ("image/png", "image/jpeg", "image/jpg", "image/bmp")
    #: Voice's uploads are audio, not images - browsers/tools report WAV
    #: under several different MIME strings, so all the common ones are
    #: accepted rather than picking one and rejecting the others.
    allowed_audio_content_types: tuple[str, ...] = ("audio/wav", "audio/wave", "audio/x-wav", "audio/vnd.wave")
    #: The hand gesture is uploaded as a JSON landmark sequence (MediaPipe runs in the browser), never as video.
    allowed_gesture_content_types: tuple[str, ...] = ("application/json",)

    #: Acceptance threshold for `template_protection.matcher.accept`'s Hamming
    #: similarity score (1.0 = identical templates, 0.0 = fully opposite).
    #: 0.9 is a conservative starting point, unvalidated against either the
    #: 128-bit or 256-bit protected-template score space; real deployments
    #: should tune this from `evaluation/privacy_metrics.py`'s EER output on
    #: their own enrolled population rather than trusting this default
    #: blindly. Left unchanged by the 128->256-bit `template_bits` migration
    #: (see that field's docstring) - this value was not re-derived from the
    #: 256-bit re-evaluation; see docs/AUTHENTICATION_RELIABILITY_REPORT.md
    #: for why calibration was deliberately deferred.
    match_threshold: float = 0.9

    #: Face decision threshold on the CALIBRATED ESTIMATE of the embedding cosine similarity
    #: (template_protection/metric_estimation.py): estimated cosine >= this -> face matches (higher = better).
    #: A teacher-requested project setting, NOT derived from genuine/impostor data - no FAR/FRR is claimed for it.
    #: The comparison itself stays a Hamming comparison: this value is translated into the equivalent
    #: template Hamming similarity through the calibration curve. Env: FACE_COSINE_THRESHOLD.
    face_cosine_threshold: float = Field(default=0.80, ge=-1.0, le=1.0)

    #: Voice decision threshold on the CALIBRATED ESTIMATE of the Euclidean distance between the unit-length
    #: voice embeddings: estimated distance <= this -> voice matches (LOWER = better; range 0..2).
    #: 0.90 (changed 2026-09-30 from the teacher-requested 0.75): on the VoxCeleb1 test split 0.75 rejects 24% of
    #: genuine clean attempts and 60-72% under everyday noise, while no impostor came closer than 0.75 even in noise.
    #: At 0.90: full set FRR 8.3% / FAR 0.96% (threshold_sensitivity.csv); in noise FRR roughly one third lower
    #: (voice_noise_frontend.csv). Chosen from these sweeps, not from a separate validation split - see
    #: docs/VOICE_MODEL.md "Threshold change to 0.90". Env: VOICE_EUCLIDEAN_THRESHOLD.
    voice_euclidean_threshold: float = Field(default=0.90, ge=0.0, le=2.0)

    #: Provenance of face_cosine_threshold / voice_euclidean_threshold. TEACHER_REQUESTED_BASELINE = the values above
    #: were set as a project requirement, not selected from genuine/impostor data (no optimality is claimed). Only a
    #: threshold selected on a validation split and documented in evaluation/reports/THRESHOLD_ANALYSIS.md may be
    #: labelled EXPERIMENTALLY_SELECTED. Informational - never changes a decision. Env: THRESHOLD_SOURCE.
    #: Note: since 2026-09-30 only the FACE value is the teacher-requested baseline; voice 0.90 was chosen from the
    #: VoxCeleb sweeps (not a held-out split), so neither label fits it exactly - see docs/VOICE_MODEL.md.
    threshold_source: Literal["TEACHER_REQUESTED_BASELINE", "EXPERIMENTALLY_SELECTED"] = "TEACHER_REQUESTED_BASELINE"

    #: Dynamic hand gesture - the Z (backend/services/hand_service.py): accept when the median DTW distance between the
    #: live gesture and the three enrolled samples is <= this value (LOWER is better). DEVELOPMENT DEFAULT: the EER
    #: threshold of the median rule on the DEV half of a SYNTHETIC calibration (tests/hand_signals.py users; see
    #: docs/HAND_GESTURE_MODALITY.md, `python -m evaluation.hand_gesture_evaluation --synthetic 40`). Synthetic users are
    #: not real people: no real-world FAR/FRR is claimed. Re-calibrate on consented real recordings with
    #: evaluation/hand_gesture_evaluation.py and then set HAND_GESTURE_THRESHOLD_SOURCE=EXPERIMENTALLY_CALIBRATED.
    #: Env: HAND_GESTURE_DTW_THRESHOLD.
    hand_gesture_dtw_threshold: float = Field(default=0.216, gt=0.0)
    #: Provenance of hand_gesture_dtw_threshold (informational - never changes a decision). Env: HAND_GESTURE_THRESHOLD_SOURCE.
    hand_gesture_threshold_source: Literal["DEVELOPMENT_DEFAULT", "EXPERIMENTALLY_CALIBRATED"] = "DEVELOPMENT_DEFAULT"
    #: Independent gesture SAMPLES required at enrollment - separate performances, each kept as its own sequence
    #: (authentication always uses ONE sample). Env: HAND_GESTURE_ENROLLMENT_ATTEMPTS (name kept for compatibility).
    hand_gesture_enrollment_attempts: int = Field(default=3, ge=3, le=10)
    #: How the distances to the enrolled samples are combined: median (default: 2 of the 3 samples must agree - robust
    #: to one poor sample, and not decided by a single lucky alignment like `min`), mean, min. Env: HAND_GESTURE_AGGREGATION.
    hand_gesture_aggregation: Literal["median", "mean", "min"] = "median"
    #: Sakoe-Chiba band of the DTW alignment, as a fraction of the sequence length. Env: HAND_GESTURE_DTW_BAND.
    hand_gesture_dtw_band: float = Field(default=0.2, gt=0.0, le=1.0)
    #: Small camera / hand rotations: the live gesture is also compared rotated by up to +/- this many degrees (in
    #: HAND_ROTATION_STEP_DEG steps) and the closest alignment counts. Deliberately small - a fully rotation-invariant
    #: matcher would discard how the user orients the Z. 0 disables the search. Env: HAND_GESTURE_ROTATION_TOLERANCE_DEG.
    hand_gesture_rotation_tolerance_deg: float = Field(default=12.0, ge=0.0, le=30.0)

    #: Security response to a SUSPICIOUS authentication attempt (backend/services/failure_policy.py - never to a single
    #: failed login): retire the ACTIVE template set and promote the next STANDBY set - every enrolled modality together
    #: (T1 -> T2). The promoted set was generated from the legitimate enrollment, under new keys; the failed
    #: capture is never stored or templated. Off by default: anyone who knows a user_id can present repeated
    #: mismatches, so each escalation spends one STANDBY set (at pool exhaustion nothing changes and the ACTIVE set
    #: keeps working). Env: TEMPLATE_ROTATION_ON_FAILED_AUTH.
    template_rotation_on_failed_auth: bool = False

    #: Failure handling (backend/services/failure_policy.py). A HIGH-QUALITY verification mismatch of one modality (a
    #: usable capture that did not match) is answered with RETRY this many times; the next consecutive one on the same
    #: ACTIVE set is a SUSPICIOUS attempt and invokes the security response above. 2 = the first attempt plus two
    #: retries, escalating on the third consecutive mismatch. Capture errors and captures failing a quality gate never
    #: count. A security policy value, not derived from data - review it for a deployment. Env: MAX_MODALITY_RETRIES.
    max_modality_retries: int = Field(default=2, ge=0)
    #: Mismatches (and unusable captures) older than this (seconds) no longer count. Env: SUSPICIOUS_MISMATCH_WINDOW_SECONDS.
    suspicious_mismatch_window_seconds: int = Field(default=900, ge=1)
    #: Unusable captures never count as mismatches and never rotate templates. They are bounded separately: after this
    #: many consecutive attempts with an unusable capture (within the window above), further attempts are refused for
    #: `capture_failure_cooldown_seconds` (HTTP 429 CAPTURE_COOLDOWN), then one attempt per cooldown until a capture is
    #: usable. A usability/rate-limit policy value, not derived from data - review it per deployment. 0 disables it.
    #: Env: MAX_CONSECUTIVE_CAPTURE_FAILURES, CAPTURE_FAILURE_COOLDOWN_SECONDS.
    max_consecutive_capture_failures: int = Field(default=5, ge=0)
    capture_failure_cooldown_seconds: int = Field(default=60, ge=1)

    #: Voice capture-quality gate (backend/services/voice_quality.py), applied to every recording at enrollment and at
    #: authentication before it is embedded. A failing recording is a capture-quality failure (retry), never a mismatch.
    #: Chosen from the VoxCeleb1 test split (evaluation/results/voice_capture_pipeline.csv, docs/VOICE_MODEL.md).
    #: Env: VOICE_MIN_SPEECH_SECONDS, VOICE_MIN_ESTIMATED_SNR_DB, VOICE_MAX_CLIPPING_FRACTION.
    #: 1.5 s: FRR at 0.75 is 78.5% for 1.5 s probes and 90.5% for 1 s (REAL DATA) - below this a sentence is not usable.
    voice_min_speech_seconds: float = Field(default=1.5, ge=0.0)
    #: 10 dB (changed 2026-09-30 from 13 dB, which rejected quiet but normal live speech, e.g. 9-12 dB in a room).
    #: REAL DATA (voice_capture_pipeline.csv): rejects 0% of clean clips and 10% of clips with white noise at 5 dB SNR
    #: (13 dB rejected 99.3%). The gate never raised the accuracy of accepted recordings; the trade-off is that more
    #: severely noisy recordings are now compared (and may mismatch) instead of being asked to re-record.
    voice_min_estimated_snr_db: float = Field(default=10.0)
    #: Conventional engineering limit (no clipped clips exist in the dataset to calibrate it on).
    voice_max_clipping_fraction: float = Field(default=0.01, ge=0.0, le=1.0)

    #: Testing aid: whether an already-named user's display name may be edited (POST /user/{id}/display-name). The
    #: display name is a label only - never part of key derivation - so editing it never touches a template. Naming a
    #: user who has no name yet is always allowed. Unset = allowed outside production, refused in production.
    #: Env: ALLOW_PROFILE_EDITING.
    allow_profile_editing: bool | None = None

    @property
    def profile_editing_enabled(self) -> bool:
        return self.allow_profile_editing if self.allow_profile_editing is not None else not self.is_production

    @field_validator("face_alignment", mode="before")
    @classmethod
    def _face_alignment_aliases(cls, value):
        return {"FACE_BASELINE": "bbox", "FACE_ALIGNED": "similarity"}.get(value, value)

    @property
    def resolved_database_url(self) -> str:
        """The SQLAlchemy URL `backend/database/session.py::get_engine` should actually use.

        `database_path` (a bare file path) takes priority over `database_url`
        (a full SQLAlchemy URL) when both are set - see `database_path`'s
        docstring for why a deployment platform is given the simpler option.
        """
        if self.database_path:
            return f"sqlite:///{self.database_path}"
        return self.database_url

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance for FastAPI's `Depends(get_settings)`.

    `lru_cache` means `.env` is only read once per process; tests that need a
    different `MASTER_SECRET`/database mid-run should call
    `get_settings.cache_clear()` after `monkeypatch.setenv(...)` (see
    tests/conftest.py).
    """
    return Settings()
