// Visual summaries derived ONLY from data the browser already captured - never simulated. Used for the gesture replay /
// thumbnails and the voiceprint strips. They are display aids, not biometric features (the backend computes those).

import type { GestureCapturePayload } from '../hand/landmarker'

// Palm centre = wrist + four finger MCPs (the same reference the backend uses for the trajectory).
const PALM = [0, 5, 9, 13, 17]

/**
 * The palm-centre trajectory of a captured gesture as an SVG path in a 100 x 100 box, mirrored like the camera
 * preview (so it looks the way the user drew it) and scaled uniformly to fit. Null when fewer than two frames had a
 * hand.
 */
export function gestureTrajectoryPath(payload: GestureCapturePayload): string | null {
  const aspect = payload.image_width / Math.max(1, payload.image_height)
  const points: [number, number][] = []
  for (const frame of payload.frames) {
    if (!frame.landmarks) continue
    const x = PALM.reduce((s, i) => s + frame.landmarks![i][0], 0) / PALM.length
    const y = PALM.reduce((s, i) => s + frame.landmarks![i][1], 0) / PALM.length
    points.push([(1 - x) * aspect, y]) // mirror + isotropic units
  }
  return fitPath(points)
}

/** The same, from the live overlay's trail (normalized video coordinates, not yet mirrored). */
export function trailPath(trail: [number, number][], aspect: number): string | null {
  return fitPath(trail.map(([x, y]) => [(1 - x) * aspect, y]))
}

function fitPath(points: [number, number][]): string | null {
  if (points.length < 2) return null
  const xs = points.map((p) => p[0])
  const ys = points.map((p) => p[1])
  const minX = Math.min(...xs)
  const minY = Math.min(...ys)
  const span = Math.max(Math.max(...xs) - minX, Math.max(...ys) - minY, 1e-6)
  const pad = 8
  const scale = (100 - 2 * pad) / span
  const offX = (100 - (Math.max(...xs) - minX) * scale) / 2
  const offY = (100 - (Math.max(...ys) - minY) * scale) / 2
  return points
    .map(([x, y], i) => `${i ? 'L' : 'M'}${((x - minX) * scale + offX).toFixed(1)} ${((y - minY) * scale + offY).toFixed(1)}`)
    .join(' ')
}

/**
 * The loudness envelope of an accepted recording: `bins` RMS values (0..1, normalized to the recording's own peak)
 * decoded from the actual WAV. Null if the browser cannot decode it.
 */
export async function voiceprintLevels(blob: Blob, bins = 48): Promise<number[] | null> {
  try {
    const context = new OfflineAudioContext(1, 1, 16000)
    const audio = await context.decodeAudioData(await blob.arrayBuffer())
    const data = audio.getChannelData(0)
    const size = Math.max(1, Math.floor(data.length / bins))
    const levels: number[] = []
    for (let b = 0; b < bins; b++) {
      let sum = 0
      const start = b * size
      const end = Math.min(data.length, start + size)
      for (let i = start; i < end; i++) sum += data[i] * data[i]
      levels.push(Math.sqrt(sum / Math.max(1, end - start)))
    }
    const peak = Math.max(...levels, 1e-6)
    return levels.map((v) => v / peak)
  } catch {
    return null
  }
}
