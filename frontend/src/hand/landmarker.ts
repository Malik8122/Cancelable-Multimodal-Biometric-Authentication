// MediaPipe Hands (HandLandmarker, VIDEO mode) - loaded once, lazily, and shared by every hand capture component.
// The model is used as-is (not trained here). Each processed video frame yields 21 landmarks (x, y normalized to the
// frame, z relative depth) for at most one hand.

import type { HandLandmarker } from '@mediapipe/tasks-vision'
import {
  GESTURE_TYPE,
  HAND_GPU_INIT_TIMEOUT_MS,
  HAND_MODEL_CDN_URL,
  HAND_MODEL_URL,
  MEDIAPIPE_WASM_CDN_URL,
  MEDIAPIPE_WASM_URL,
} from '../config/hand'

let loading: Promise<HandLandmarker> | null = null

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('timed out')), ms)
    promise.then(
      (value) => (clearTimeout(timer), resolve(value)),
      (error) => (clearTimeout(timer), reject(error)),
    )
  })
}

async function create(wasmUrl: string, modelUrl: string): Promise<HandLandmarker> {
  const { FilesetResolver, HandLandmarker } = await import('@mediapipe/tasks-vision')
  const fileset = await FilesetResolver.forVisionTasks(wasmUrl)
  const options = {
    baseOptions: { modelAssetPath: modelUrl },
    runningMode: 'VIDEO' as const,
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  }
  try {
    // Some GPU drivers hang here instead of failing - fall back to the CPU after a timeout.
    return await withTimeout(
      HandLandmarker.createFromOptions(fileset, { ...options, baseOptions: { ...options.baseOptions, delegate: 'GPU' } }),
      HAND_GPU_INIT_TIMEOUT_MS,
    )
  } catch {
    return await HandLandmarker.createFromOptions(fileset, { ...options, baseOptions: { ...options.baseOptions, delegate: 'CPU' } })
  }
}

export function loadHandLandmarker(): Promise<HandLandmarker> {
  if (!loading) {
    loading = create(MEDIAPIPE_WASM_URL, HAND_MODEL_URL)
      // Self-hosted files missing (e.g. the copy script could not download the model): use the CDNs.
      .catch(() => create(MEDIAPIPE_WASM_CDN_URL, HAND_MODEL_CDN_URL))
      .catch((error) => {
        loading = null // allow a retry after a network failure
        throw error
      })
  }
  return loading
}

/** One frame of the wire format (backend preprocessing/hand_gesture.py::parse_capture). */
export interface GestureFrame {
  t: number
  landmarks: number[][] | null
  handedness?: string
}

/** One gesture sample as uploaded: landmarks only, never video. */
export interface GestureCapturePayload {
  gesture_type: string
  /** The user drew while watching a mirrored (selfie) preview; the backend flips x back to the user's left/right. */
  mirrored: boolean
  image_width: number
  image_height: number
  frames: GestureFrame[]
  client_metrics: { mediapipe_ms_mean: number; mediapipe_ms_max: number; fps: number }
}

export function toPayload(frames: GestureFrame[], width: number, height: number, inferenceMs: number[]): GestureCapturePayload {
  const duration = frames.length > 1 ? frames[frames.length - 1].t - frames[0].t : 0
  const mean = inferenceMs.length ? inferenceMs.reduce((a, b) => a + b, 0) / inferenceMs.length : 0
  return {
    gesture_type: GESTURE_TYPE,
    mirrored: true,
    image_width: width,
    image_height: height,
    frames,
    client_metrics: {
      mediapipe_ms_mean: Math.round(mean * 100) / 100,
      mediapipe_ms_max: Math.round(Math.max(0, ...inferenceMs) * 100) / 100,
      fps: duration > 0 ? Math.round(((frames.length - 1) * 1000 * 10) / duration) / 10 : 0,
    },
  }
}

export function payloadBlob(payload: GestureCapturePayload): Blob {
  return new Blob([JSON.stringify(payload)], { type: 'application/json' })
}
