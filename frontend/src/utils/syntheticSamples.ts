import { encodeWav } from './wav'

// These build the same kind of synthetic, non-biometric samples the
// backend's own offline test suite uses (see tests/voice_signals.py and
// tests/test_fusion_endpoint.py's _tone()) - good enough for a real voice
// enroll/verify roundtrip against the real checkpoint without a microphone.
// The image is a plain stripe pattern with no face in it: it is only ever sent
// for a modality that is not enrolled (ENROLLMENT_REQUIRED, never decoded).
// The Testing Mode page uses these to run real API calls, not to fabricate
// results - every score/decision shown still comes back from the live
// backend.

export function syntheticImagePng(): Promise<Blob> {
  const size = 300
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  const image = ctx.createImageData(size, size)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const value = Math.round(Math.sin((x / size) * 20 * Math.PI) * 127 + 128)
      const offset = (y * size + x) * 4
      image.data[offset] = value
      image.data[offset + 1] = value
      image.data[offset + 2] = value
      image.data[offset + 3] = 255
    }
  }
  ctx.putImageData(image, 0, 0)
  return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob!), 'image/png'))
}

export function syntheticToneWav(durationSeconds = 2, frequency = 220, sampleRate = 16_000): Blob {
  const length = Math.floor(sampleRate * durationSeconds)
  const samples = new Float32Array(length)
  for (let i = 0; i < length; i++) {
    // 4 Hz loudness envelope (10%-100%) so the backend's voice capture-quality gate sees speech-like level variation;
    // a pure tone reads as all noise and is (correctly) asked to be re-recorded. See tests/voice_signals.py.
    const envelope = 0.1 + 0.9 * (0.5 + 0.5 * Math.cos((2 * Math.PI * 4 * i) / sampleRate))
    samples[i] = 0.3 * envelope * Math.sin((2 * Math.PI * frequency * i) / sampleRate)
  }
  return encodeWav(samples, sampleRate)
}
