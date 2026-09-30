// Mirrors backend/database/schema.py exactly. Nothing here is computed on
// the frontend - these are wire types for what the backend already returns.

export type Modality = 'face' | 'voice' | 'hand'

export interface EnrollResponse {
  success: boolean
  user_id: string
  modality: Modality
  /** How many template sets this modality's templates were written into. */
  templates_created: number
  active_template_set_version: number
  standby_template_set_versions: number[]
  template_version: number
  key_version: number
  template_id: string
  /** Face five-pose enrollment: how many poses were valid (the centroid was averaged over those) and each pose's verdict. */
  poses_valid?: number
  pose_results?: PoseResult[]
  /** Voice with two recordings: EXCELLENT / GOOD, or FAIR when the user chose to continue. */
  recording_quality?: RecordingQuality
  /** Hand gesture enrollment: gesture type, attempts, frames and duration per attempt (never landmarks). */
  hand_gesture?: {
    gesture_type: string
    attempts: number
    attempt_frames_detected: number[]
    attempt_durations_s: number[]
    template_format: string
  }
}

/** POST /enroll/hand/check: the backend's capture-gate verdict for one gesture sample. */
export interface HandGestureCheckResponse {
  attempt?: number
  /** OK / NO_HAND_DETECTED / INSUFFICIENT_FRAMES / CAPTURE_QUALITY_FAILURE / INVALID_TRAJECTORY / MALFORMED_CAPTURE */
  verdict: string
  passed: boolean
  message: string
  metrics: Record<string, number>
}

/** The hand gesture part of an authentication response. Distances only with DEBUG_SCORES on the backend. */
export interface HandGestureOutcome {
  /** VERIFIED / VERIFICATION_MISMATCH / QUALITY_INSUFFICIENT / CAPTURE_ERROR */
  status: string
  /** MATCH / GESTURE_MISMATCH, or a capture failure code (e.g. NO_HAND_DETECTED). */
  decision: string
  gesture_type?: string
  frames_total?: number
  frames_detected?: number
  duration_s?: number
  timing_ms?: Record<string, number>
  reason?: string
  dtw_distance?: number
  dtw_distances?: number[]
  threshold?: number
  threshold_source?: string
  tracking_fps?: number
  motion_duration_s?: number
  idle_trimmed_s?: number
  /** Resampled trajectory length, Z-structure verdict, and the per-sample aggregation rule (median / mean / min). */
  trajectory_points?: number
  z_validity?: string
  aggregation?: string
  /** Development: rotation (deg) behind each per-sample distance; each feature group's share of that distance. */
  dtw_best_rotation_deg?: number[]
  feature_group_shares?: Record<string, number>[]
  /** Tempo diagnostics: movement time / enrolled median, palm path length and mean speed, time share per Z stroke. */
  normalized_duration?: number
  path_length_palms?: number
  mean_speed_palms_per_s?: number
  segment_time_fractions?: number[]
  /** Development only: per-finger shares, fingertip extension live vs enrolled, shadow distances (never decisive). */
  finger_shares?: Record<string, number>[]
  finger_extension_live?: Record<string, number>
  finger_extension_enrolled?: Record<string, number>[]
  shadow_distances?: Record<string, { per_sample: number[]; median: number }>
}

// Face enrollment is one guided five-capture protocol (mostly-frontal, not head turns - see
// GuidedFaceCapture.tsx). Per capture the backend runs MTCNN + alignment + quality gating and
// rejects a capture that has no face, more than one face, is blurry, or is too small/off-center/
// angled in frame, or too low-confidence a detection; the valid embeddings are averaged into a
// centroid and the templates come from it alone.
export type FacePose = 'front' | 'left' | 'right' | 'up' | 'down'
export type PoseStatus = 'VALID' | 'NO_FACE' | 'BLURRY' | 'MULTIPLE_FACES' | 'TOO_SMALL' | 'OFF_CENTER' | 'TOO_ANGLED' | 'LOW_CONFIDENCE' | 'ALIGNMENT_FAILED' | 'LANDMARK_FAILURE'

export interface PoseResult {
  pose: FacePose
  status: PoseStatus
}

export interface FacePoseCheckResponse {
  pose: FacePose
  status: PoseStatus
  valid: boolean
  detail?: string
}

export type EnrollmentStatus = 'NOT_REGISTERED' | 'REGISTERED' | 'UPDATED' | 'RETRY_REQUIRED'

/** Voice capture-quality gate verdicts (backend/services/voice_quality.py). */
export type VoiceQualityVerdict = 'OK' | 'INVALID_AUDIO' | 'NO_SPEECH' | 'CLIPPED' | 'TOO_LITTLE_SPEECH' | 'TOO_NOISY'

/** POST /voice/check-quality: the gate verdict for one recording. Scalars only - nothing is stored. */
export interface VoiceQualityCheckResponse {
  sentence?: number
  verdict: VoiceQualityVerdict
  passed: boolean
  message: string
  metrics: Record<string, number | null>
  /** DEBUG_SCORES only: levels, noise floor, speech-band SNR, input format, the limits and the failing check. */
  diagnostics?: VoiceQualityDiagnostics
}

export interface VoiceQualityDiagnostics {
  input_sample_rate_hz?: number
  channels?: number
  input_duration_seconds?: number
  rms_dbfs?: number | null
  peak_dbfs?: number | null
  speech_level_dbfs?: number | null
  noise_floor_dbfs?: number | null
  silence_seconds?: number | null
  low_frequency_energy_ratio?: number | null
  speech_band_snr_db?: number | null
  limits: { min_speech_seconds: number; min_estimated_snr_db: number; max_clipping_fraction: number }
  failed_check: { metric: string; required: string; measured: number | null } | null
}

// Voice enrollment: how consistent the two recordings are (ECAPA embedding cosine similarity, computed by the backend).
//   EXCELLENT / GOOD  -> enrolled.   FAIR -> LOW_QUALITY_WARNING, nothing stored until the user continues.
//   POOR              -> ENROLLMENT_INCONSISTENT, rejected, nothing stored.
export type RecordingQuality = 'EXCELLENT' | 'GOOD' | 'FAIR' | 'POOR'

// HTTP 409 body of /enroll (voice): usable but lower-quality recordings. Nothing was stored yet.
export interface LowQualityWarningResponse {
  status: 'LOW_QUALITY_WARNING'
  recording_quality: 'FAIR'
  detail: string
  modality: Modality
}

// HTTP 422 body of /enroll when the two voice recordings are not consistent: nothing was stored.
export interface EnrollmentInconsistentResponse {
  status: 'ENROLLMENT_INCONSISTENT'
  recording_quality: 'POOR'
  detail: string
  modality: Modality
  enrollment_status: 'RETRY_REQUIRED'
}

export type FusionPolicy = 'ALL_REQUIRED' | 'AT_LEAST_TWO' | 'WEIGHTED'

// The ONE authentication response (/verify/*, /authenticate, /authenticate/fusion):
// one decision, one fusion similarity, one template set version. Per-modality
// similarity / distance / threshold values never reach the frontend - the
// backend only returns them when it runs with DEBUG_SCORES=true, and this app
// deliberately has no field for them.
export type AuthenticationState = 'ACCESS_GRANTED' | 'ACCESS_DENIED' | 'ENROLLMENT_REQUIRED'

/**
 * Backend security response to a SUSPICIOUS attempt (backend/services/authentication.py::rotate_after_failed_authentication):
 * the ACTIVE template set is retired and the next STANDBY set - generated at enrollment from the user's own
 * biometrics - becomes ACTIVE for every enrolled modality together. Versions and modality names only; never keys.
 */
export interface TemplateRotation {
  status: 'ROTATED' | 'ALREADY_ROTATED' | 'POOL_EXHAUSTED' | 'ROTATION_FAILED'
  triggered: boolean
  active_set_before: number
  active_set_after: number
  modalities: Modality[]
  remaining_standby_sets: number
  reason: string
}

/** Per-modality verification status inside one attempt (backend/states.py). */
export type ModalityVerificationStatus = 'VERIFIED' | 'VERIFICATION_MISMATCH' | 'QUALITY_INSUFFICIENT' | 'CAPTURE_ERROR'

/**
 * How the backend handled a DENIED attempt (backend/services/failure_policy.py) - displayed as-is, never recomputed.
 * RETRY_REQUESTED: capture error, insufficient quality or an isolated mismatch - no template change.
 * SUSPICIOUS_ATTEMPT: repeated high-quality mismatches - the security response (template rotation) was invoked.
 */
export interface AttemptAssessment {
  status: 'RETRY_REQUESTED' | 'SUSPICIOUS_ATTEMPT'
  modality_statuses: Partial<Record<Modality, ModalityVerificationStatus>>
  /** Presented modalities to capture again (empty after an escalation). */
  retry_modalities: Modality[]
  retries_remaining: number
  headline: string
  reason: string
  modality_messages: Partial<Record<Modality, string>>
  /** Why a capture was unusable, per modality (e.g. "Sentence 2: too much background noise ..."). */
  modality_hints?: Partial<Record<Modality, string>>
  /** Consecutive high-quality mismatch number (0 = none) out of the number that escalates, e.g. 1 of 3. */
  mismatch_attempt?: number
  mismatch_attempt_limit?: number
  capture_failures?: number
  capture_failure_limit?: number
  /** The attempt numbers in words (backend wording). */
  attempt_notice?: string
  security_response: string
}

export interface AuthenticationDecision {
  user_id: string
  /** ACCESS_GRANTED or ACCESS_DENIED - an evaluation took place. */
  authentication_state: 'ACCESS_GRANTED' | 'ACCESS_DENIED'
  /** Same value as authentication_state (backend compatibility field). */
  status: 'ACCESS_GRANTED' | 'ACCESS_DENIED'
  authenticated: boolean
  fusion_similarity: number
  fusion_distance: number
  fusion_threshold: number
  fusion_policy: FusionPolicy
  matched_modalities: Modality[]
  modalities_used: Modality[]
  /** The ACTIVE template set that was matched. */
  active_template_set: number
  template_set_version: number
  /** Highest HKDF key version among the modalities' templates in that set. */
  key_version: number
  authentication_time_ms: number
  /** The facility label of the session (context only). */
  building_id?: string
  /** The user's human-readable name - sent only with ACCESS_GRANTED. */
  display_name?: string
  /** Present only after a SUSPICIOUS_ATTEMPT escalation, when the backend has template rotation enabled. */
  template_rotation?: TemplateRotation
  /** Present only on ACCESS_DENIED: retry guidance, or the suspicious-attempt escalation. */
  attempt_assessment?: AttemptAssessment
  /** Present when the hand gesture was submitted. */
  hand_gesture?: HandGestureOutcome
  /**
   * Per-modality results - present only when the backend runs with DEBUG_SCORES=true
   * (backend/services/authentication.py::to_public_response). Never an embedding, template or key.
   */
  results?: Partial<Record<Modality, ModalityAuthenticationResult>>
  /**
   * Local-development observability only - present exclusively when the backend runs with
   * DEBUG_SCORES=true (see backend/services/authentication.py::to_public_response and
   * fusion/diagnostics.py::build_fusion_diagnostics). Absent in production responses.
   */
  fusion_diagnostics?: FusionDiagnostics
}

/**
 * Face: 'cosine_estimate' (higher = better). Voice: 'euclidean_estimate' (LOWER = better). Both are calibrated
 * ESTIMATES derived from the template Hamming comparison (template_protection/metric_estimation.py), not exact
 * embedding metrics. Other modalities: 'hamming'.
 */
export type ModalityMetric = 'cosine_estimate' | 'euclidean_estimate' | 'hamming'

export interface ModalityAuthenticationResult {
  /** Fusion-scale score (estimated cosine, higher = better) and its threshold. */
  score: number
  threshold: number
  authenticated: boolean
  metric: ModalityMetric
  metric_value: number
  metric_threshold: number
  metric_higher_is_better: boolean
  metric_uncertainty: number
  /** The cancelable-template comparison itself. */
  hamming_similarity: number
  hamming_distance_bits: number
  template_bits: number
}

export type FusionModalityStatus = 'verified' | 'failed_below_threshold' | 'not_presented' | 'capture_error'

export interface FusionModalityDiagnostics {
  /** Hamming similarity in [0, 1], or null when this modality was not presented. */
  score: number | null
  /** The per-modality threshold that score was compared against, or null when not presented. */
  threshold: number | null
  verified: boolean | null
  status: FusionModalityStatus
}

export interface FusionDiagnostics {
  face: FusionModalityDiagnostics
  voice: FusionModalityDiagnostics
  hand?: FusionModalityDiagnostics
  /** Fusion weight actually applied to each presented modality (fusion/score_fusion.py) - sums to 1. */
  weights: Partial<Record<Modality, number>>
  /** fusion/score_fusion.py::fuse_scores's real output - the same value as fusion_similarity/fused_score above. */
  fused_score: number
  threshold: number
  policy: FusionPolicy
  /** The backend's real access decision (same as `authenticated` above) - under ALL_REQUIRED this
   * is NOT simply fused_score >= threshold; one failed modality denies access regardless. */
  access_granted: boolean
}

// HTTP 409 body of /authenticate/fusion: a modality the user SUBMITTED is not enrolled.
// This is NOT an authentication failure - nothing biometric was evaluated.
export interface EnrollmentRequiredResponse {
  status: 'ENROLLMENT_REQUIRED'
  authentication_state: 'ENROLLMENT_REQUIRED'
  detail: string
  user_id: string
  building_id?: string
  submitted_modalities: Modality[]
  enrolled_modalities: Modality[]
  missing_modalities: Modality[]
}

/** What the result screen renders: exactly one of the three authentication states. */
export type AuthenticationOutcome = AuthenticationDecision | EnrollmentRequiredResponse

// --- Buildings (authentication context only) + the user's enrollment profile ---
export interface BuildingInfo {
  id: string
  name: string
  description: string
  clearance_level: string
}

export interface EnrollmentStatusResponse {
  user_id: string
  application_id: string
  modalities: Partial<Record<Modality, boolean>>
  statuses: Partial<Record<Modality, EnrollmentStatus>>
  /** The stored name, or the "User <short id>" fallback when has_display_name is false. */
  display_name: string
  has_display_name: boolean
  /** Whether an existing display name may be edited (testing aid; off in production). */
  profile_editing_enabled?: boolean
}

export interface UserProfileResponse {
  user_id: string
  display_name: string
}

export type AuthenticateResponse = AuthenticationDecision
export type FusionAuthenticateResponse = AuthenticationDecision

export type TemplateSetStatus = 'ACTIVE' | 'STANDBY' | 'REVOKED'

export interface RevokeResponse {
  success: boolean
  user_id: string
  revoked_template_set_version: number
  new_active_template_set_version: number
  remaining_standby_template_sets: number
}

export interface TemplateSetInfo {
  template_set_version: number
  status: TemplateSetStatus
  modalities: Modality[]
  key_versions: Partial<Record<Modality, number>>
  template_group_id: string | null
  created_at: string | null
  activated_at: string | null
  revoked_at: string | null
  revoked_reason: string | null
}

export interface TemplateSetPoolResponse {
  user_id: string
  application_id: string
  pool_size: number
  active_template_set_version: number | null
  standby_count: number
  sets: TemplateSetInfo[]
}

export interface ActivateResponse {
  success: boolean
  user_id: string
  previous_active_template_set_version: number | null
  new_active_template_set_version: number
  remaining_standby_template_sets: number
}

export interface GenerateSetResponse {
  success: boolean
  user_id: string
  new_template_set_version: number
  standby_template_set_versions: number[]
  active_template_set_version: number
}

export interface EnrolledModality {
  modality: Modality
  application_id: string
  template_version: number
  key_version: number
  created_at: string
}

export interface UserModalitiesResponse {
  user_id: string
  enrolled_modalities: EnrolledModality[]
}

export interface ModalityMetricsResponse {
  modality: Modality
  available: boolean
  calibrated: boolean
  metrics: Partial<{
    num_samples: number
    num_genuine_pairs: number
    num_impostor_pairs: number
    eer: number
    eer_threshold: number
    accuracy: number
    far: number
    frr: number
    precision: number
    recall: number
    f1: number
    auc: number
    // Real protected-template calibration (evaluation/threshold_calibration.py) -
    // a different score space than the raw-embedding fields above, so these
    // are deliberately separate keys, never merged into far/frr/eer/auc.
    calibrated_threshold: number
    calibrated_far: number
    calibrated_frr: number
    calibrated_eer: number
    calibrated_auc: number
  }>
}

export interface SystemHealthResponse {
  backend: string
  database: string
  face_model: string
  voice_model: string
  template_protection: string
  fusion_policy: FusionPolicy
  thresholds_loaded: boolean
  audit_logging: boolean
  template_pool_size: number
  /** The backend runs with DEBUG_SCORES: show capture / matching diagnostics. */
  diagnostics_enabled?: boolean
}

export interface AuditLogEntry {
  audit_id: string
  timestamp: string
  user_id: string
  building_id: string | null
  modality_list: Modality[]
  fusion_similarity: number | null
  fusion_policy: FusionPolicy | null
  authentication_state: AuthenticationState
  submitted_modalities: Modality[] | null
  enrolled_modalities: Modality[] | null
  authenticated_modalities: Modality[] | null
  authenticated: boolean
  latency_ms: number
  template_versions: Partial<Record<Modality, number>>
  key_versions: Partial<Record<Modality, number>>
  /** Failure handling (null on rows written before it existed). */
  modality_statuses?: Partial<Record<Modality, ModalityVerificationStatus>> | null
  attempt_status?: 'ACCESS_GRANTED' | 'RETRY_REQUESTED' | 'SUSPICIOUS_ATTEMPT' | null
  template_rotation_status?: string | null
}

export interface AuditHistoryResponse {
  user_id: string
  total: number
  limit: number
  offset: number
  entries: AuditLogEntry[]
}

export interface SystemAuditResponse {
  entries: AuditLogEntry[]
}

export interface ApiErrorBody {
  detail: string
}

// Not a backend type - thrown by api/client.ts for any non-2xx response so
// callers can distinguish "the backend said no" from "the backend is down".
export class ApiError extends Error {
  status: number
  detail: string
  /** The parsed JSON error body, when there was one (e.g. the ENROLLMENT_REQUIRED 409). */
  body?: unknown

  constructor(status: number, detail: string, body?: unknown) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
    this.body = body
  }
}
