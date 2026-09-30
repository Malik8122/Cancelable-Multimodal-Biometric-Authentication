"""Shared vocabularies: authentication outcomes and per-modality enrollment status.

Authentication outcomes (the same names in the API, audit log, frontend and tests):
- ACCESS_GRANTED: every submitted modality verified.
- ACCESS_DENIED: every submitted modality was enrolled, but at least one failed verification.
- ENROLLMENT_REQUIRED: a submitted modality is not enrolled. NOT an authentication failure - nothing was verified.

Enrollment status of one modality:
- NOT_REGISTERED: nothing enrolled.
- REGISTERED: enrolled.
- UPDATED: enrolled, and re-enrolled at least once (the earlier templates were replaced).
- RETRY_REQUIRED (voice only): the last enrollment attempt failed the two-recording consistency check, nothing was stored.
"""

ACCESS_GRANTED = "ACCESS_GRANTED"
ACCESS_DENIED = "ACCESS_DENIED"
ENROLLMENT_REQUIRED = "ENROLLMENT_REQUIRED"
ENROLLMENT_INCONSISTENT = "ENROLLMENT_INCONSISTENT"
#: Voice: the recordings are usable but of lower quality - nothing is stored until the user chooses to continue.
LOW_QUALITY_WARNING = "LOW_QUALITY_WARNING"

AUTHENTICATION_STATES = (ACCESS_GRANTED, ACCESS_DENIED, ENROLLMENT_REQUIRED)

# Per-modality verification status inside one attempt (backend/services/failure_policy.py). Only VERIFICATION_MISMATCH -
# a usable capture that did not match - counts towards a suspicious authentication attempt; an unusable capture never does.
VERIFIED = "VERIFIED"
VERIFICATION_MISMATCH = "VERIFICATION_MISMATCH"
#: The capture was processed, but fails the modality's existing quality gate (face: the enrollment gates).
QUALITY_INSUFFICIENT = "QUALITY_INSUFFICIENT"
#: The sample could not be processed at all (no face detected, undecodable or silent audio, unreadable image).
CAPTURE_ERROR = "CAPTURE_ERROR"
MODALITY_VERIFICATION_STATUSES = (VERIFIED, VERIFICATION_MISMATCH, QUALITY_INSUFFICIENT, CAPTURE_ERROR)

# Security response to one attempt (ACCESS_DENIED is still the decision for both of the last two).
#: Denied, not escalated: the failed modalities should be presented again. Never rotates templates.
RETRY_REQUESTED = "RETRY_REQUESTED"
#: Denied after repeated high-quality mismatches: the existing security response (template rotation) is invoked. A
#: suspicious attempt, not a detected attacker - a biometric mismatch says nothing certain about intent.
SUSPICIOUS_ATTEMPT = "SUSPICIOUS_ATTEMPT"
#: Too many consecutive unusable captures: the attempt is refused for a short time (HTTP 429). A rate limit only - no
#: evaluation, no mismatch counted, no template change.
CAPTURE_COOLDOWN = "CAPTURE_COOLDOWN"
ATTEMPT_STATUSES = (ACCESS_GRANTED, RETRY_REQUESTED, SUSPICIOUS_ATTEMPT, CAPTURE_COOLDOWN)

NOT_REGISTERED = "NOT_REGISTERED"
REGISTERED = "REGISTERED"
UPDATED = "UPDATED"
RETRY_REQUIRED = "RETRY_REQUIRED"
ENROLLMENT_STATUSES = (NOT_REGISTERED, REGISTERED, UPDATED, RETRY_REQUIRED)

#: The modalities a user can enroll and present.
BIOMETRIC_MODALITIES = ("face", "voice", "hand")
