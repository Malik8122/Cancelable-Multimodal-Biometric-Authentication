// Dynamic hand gesture capture settings. MediaPipe Hands runs HERE, in the browser; only the resulting landmark
// sequence (never video) is sent to the backend, which validates, normalizes, and matches it (DTW).

/** The gesture this build enrolls and verifies: a handwritten-style Z (backend GESTURE_TYPES). */
export const GESTURE_TYPE = 'z'
export const GESTURE_LABEL = 'Z gesture'
/** The backend representation these captures are made for (preprocessing/hand_gesture.py HAND_TEMPLATE_FORMAT);
 *  recorded with research data so samples from another pipeline version are never mixed silently. */
export const HAND_PIPELINE_VERSION = 'hand_gesture_v2'

/**
 * The Z guide, in a 200 x 100 box as the user sees it (the preview is mirrored): top-left -> top-right -> diagonally
 * down-left -> bottom-right. A guide only - what is drawn after a capture is always the real tracked trajectory.
 */
export const Z_GUIDE_PATH = 'M55 18 L145 18 L55 82 L145 82'
/** The four corner dots of the guide (start, two turns, end). */
export const Z_GUIDE_POINTS: readonly [number, number][] = [[55, 18], [145, 18], [55, 82], [145, 82]]

/** Enrollment: three independent Z samples; authentication: one (config/protocol.ts). */
export { HAND_AUTHENTICATION_SAMPLES, HAND_ENROLLMENT_SAMPLES } from './protocol'

/** A recording stops automatically after this long (the backend accepts 0.5-10 s of hand movement). */
export const MAX_RECORDING_MS = 6000

/** Client-side hint only (the backend decides): below this many tracked frames a recording is certainly too short. */
export const MIN_TRACKED_FRAMES_HINT = 15

// MediaPipe runtime + model, self-hosted from public/mediapipe/ (scripts/copy-mediapipe.mjs, run before dev/build) so
// the tracker starts in about a second instead of downloading ~20 MB from two CDNs. The CDN URLs are the fallback.
export const MEDIAPIPE_WASM_URL = import.meta.env.VITE_MEDIAPIPE_WASM_URL ?? '/mediapipe/wasm'
export const HAND_MODEL_URL = import.meta.env.VITE_HAND_MODEL_URL ?? '/mediapipe/hand_landmarker.task'
export const MEDIAPIPE_WASM_CDN_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm'
export const HAND_MODEL_CDN_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
/** GPU start-up can hang on some drivers instead of failing; after this long the CPU is used. */
export const HAND_GPU_INIT_TIMEOUT_MS = 8000
