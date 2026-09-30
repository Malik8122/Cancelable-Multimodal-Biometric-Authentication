import type { AuthenticationDecision, Modality, ModalityVerificationStatus } from '../api/types'

const HAND_STATUSES = new Set<string>(['VERIFIED', 'VERIFICATION_MISMATCH', 'QUALITY_INSUFFICIENT', 'CAPTURE_ERROR'])

/** Presented factors in display order. */
export const FACTOR_ORDER: Modality[] = ['face', 'voice', 'hand']

/**
 * The verdict for one presented factor, taken from what the backend returned - never recomputed: the attempt
 * assessment's per-modality status when present (denied attempts), the hand gesture outcome, otherwise whether the
 * factor is among the matched modalities. No score is needed, so it works without DEBUG_SCORES.
 */
export function modalityOutcome(result: AuthenticationDecision, m: Modality): ModalityVerificationStatus {
  const assessed = result.attempt_assessment?.modality_statuses?.[m]
  if (assessed) return assessed
  if (m === 'hand' && result.hand_gesture && HAND_STATUSES.has(result.hand_gesture.status)) {
    return result.hand_gesture.status as ModalityVerificationStatus
  }
  return result.matched_modalities.includes(m) ? 'VERIFIED' : 'VERIFICATION_MISMATCH'
}

export type OutcomeTone = 'success' | 'warning' | 'danger'

export function outcomeTone(status: ModalityVerificationStatus): OutcomeTone {
  return status === 'VERIFIED' ? 'success' : status === 'QUALITY_INSUFFICIENT' || status === 'CAPTURE_ERROR' ? 'warning' : 'danger'
}
