// Self-hosts the MediaPipe hand tracker so it loads from this app instead of two CDNs on every visit:
// copies the WASM runtime from the installed @mediapipe/tasks-vision package (version always matches the JS) and
// downloads the hand_landmarker.task model once. Output: public/mediapipe/ (git-ignored). Runs before dev and build.
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const wasmSrc = join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm')
const outDir = join(root, 'public', 'mediapipe')
const wasmOut = join(outDir, 'wasm')
const modelOut = join(outDir, 'hand_landmarker.task')
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

mkdirSync(wasmOut, { recursive: true })
for (const file of readdirSync(wasmSrc)) {
  const src = join(wasmSrc, file)
  const dst = join(wasmOut, file)
  if (!existsSync(dst) || statSync(dst).size !== statSync(src).size) copyFileSync(src, dst)
}

if (!existsSync(modelOut) || statSync(modelOut).size === 0) {
  try {
    const response = await fetch(MODEL_URL)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    writeFileSync(modelOut, Buffer.from(await response.arrayBuffer()))
    console.log('[mediapipe] hand model downloaded to public/mediapipe/')
  } catch (error) {
    // Not fatal: the app falls back to the CDN model URL at runtime.
    console.warn(`[mediapipe] could not download the hand model (${error.message}); the CDN will be used instead`)
  }
}
