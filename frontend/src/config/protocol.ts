// The biometric protocols of this console - ENROLLMENT and AUTHENTICATION are separate, and nothing here is shared
// between them. Every capture count shown in the UI is derived from these values. (Pure data: no runtime imports, so
// the protocol tests in frontend/tests/ can load it directly.)

import type { Modality } from '../api/types'

// ---------------------------------------------------------------------------------------------------- registration

export interface RegistrationStep {
  /** 1-based biometric step number shown to the user. */
  step: number
  modality: Modality
  /** Registration is complete without it (the backend's enrollment status: face + voice). */
  optional: boolean
}

/**
 * Biometric registration order: 1 Face, 2 Voice, 3 Dynamic Hand Gesture. Entering the user's name comes first but is
 * account setup, not a biometric step, so it is not numbered.
 */
export const REGISTRATION_STEPS: readonly RegistrationStep[] = [
  { step: 1, modality: 'face', optional: false },
  { step: 2, modality: 'voice', optional: false },
  { step: 3, modality: 'hand', optional: true },
]

export function registrationStep(modality: Modality): RegistrationStep {
  const step = REGISTRATION_STEPS.find((s) => s.modality === modality)
  if (!step) throw new Error(`${modality} is not a registration step`)
  return step
}

export function registrationStepLabel(modality: Modality): string {
  const { step, optional } = registrationStep(modality)
  return optional ? `Step ${step} - Optional` : `Step ${step}`
}

// ------------------------------------------------------------------------------------------------------------ face

/** Face enrollment: five guided samples (GuidedFaceCapture), averaged into one centroid by the backend. */
export const FACE_ENROLLMENT_SAMPLES = 5
/** Face authentication: one capture. */
export const FACE_AUTHENTICATION_CAPTURES = 1

// ----------------------------------------------------------------------------------------------------------- voice

/**
 * Voice ENROLLMENT prompts, one recording each. The UI shows "Sentence n / N" from this list. The backend's voice
 * enrollment takes the first recording as the sample and the second as its consistency check (POST /enroll with
 * image + confirm_image), so this list must stay in step with that endpoint.
 * The speaker model is text-independent (ECAPA-TDNN): the words are never checked, they only give the user something
 * natural to say for 3-5 seconds.
 */
export const VOICE_ENROLLMENT_PROMPTS: readonly string[] = [
  'Secure access to the control room.',
  'Biometric verification is required.',
]

/**
 * Voice AUTHENTICATION: TWO sentences (POST /authenticate/fusion `voice_audio` + `voice_audio_2`). The backend embeds
 * each and combines them exactly like enrollment, then compares once. Measured at the earlier 0.75 threshold: genuine
 * rejections 12.2% with one sentence vs 2.6% with two, false accepts 0.03% vs 0.013% (docs/VOICE_MODEL.md).
 */
export const VOICE_AUTHENTICATION_PROMPTS: readonly string[] = VOICE_ENROLLMENT_PROMPTS
export const VOICE_AUTHENTICATION_PROMPT = VOICE_AUTHENTICATION_PROMPTS[0]
export const VOICE_AUTHENTICATION_SENTENCES = VOICE_AUTHENTICATION_PROMPTS.length

/** The prompts a voice capture step asks for: the enrollment prompts, or the authentication prompts. */
export function voicePrompts(mode: 'register' | 'verify'): readonly string[] {
  return mode === 'register' ? VOICE_ENROLLMENT_PROMPTS : VOICE_AUTHENTICATION_PROMPTS
}

/** One empty recording slot per prompt of the protocol. */
export function emptyVoiceSlots<T>(mode: 'register' | 'verify'): (T | null)[] {
  return voicePrompts(mode).map(() => null)
}

// ------------------------------------------------------------------------------------------------------------ hand

/**
 * Hand ENROLLMENT: three independent Z gesture samples - three separate, natural performances of the Z (their size,
 * position, speed and timing naturally differ). Each is kept as its own sequence in the protected template; the
 * backend refuses the same captured sequence twice, and a sample unlike both others is asked for again. Must match the
 * backend's HAND_GESTURE_ENROLLMENT_ATTEMPTS.
 */
export const HAND_ENROLLMENT_SAMPLES = 3
/** Hand AUTHENTICATION: one Z gesture sample. */
export const HAND_AUTHENTICATION_SAMPLES = 1

export function handSamples(mode: 'register' | 'verify'): number {
  return mode === 'register' ? HAND_ENROLLMENT_SAMPLES : HAND_AUTHENTICATION_SAMPLES
}

/** "01 / 05" */
export function sampleCounter(current: number, total: number): string {
  const width = String(total).length < 2 ? 2 : String(total).length
  return `${String(current).padStart(width, '0')} / ${String(total).padStart(width, '0')}`
}
