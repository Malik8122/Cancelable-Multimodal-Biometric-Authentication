// Voice capture configuration. The PROTOCOLS (how many sentences at enrollment vs authentication) live in
// config/protocol.ts; the recording settings below are identical for both, so every recording is made the same way.

export { VOICE_AUTHENTICATION_PROMPT, VOICE_ENROLLMENT_PROMPTS, voicePrompts } from './protocol'

/**
 * Microphone constraints, identical for enrollment and verification. The browser's own echo cancellation, noise
 * suppression and automatic gain control are switched OFF: they process each recording differently (adaptive,
 * content-dependent), while the voice model was trained on unprocessed recordings. Background noise is handled by the
 * backend's capture-quality gate instead (the user is asked to re-record). The effect of this choice on real
 * microphones has not been measured offline - see docs/VOICE_MODEL.md.
 */
export const MICROPHONE_CONSTRAINTS: MediaTrackConstraints = {
  channelCount: 1,
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
}

/** Recording length per sentence: shorter recordings are discarded, recording stops by itself at the maximum. */
export const VOICE_MIN_SECONDS = 3
export const VOICE_MAX_SECONDS = 5
