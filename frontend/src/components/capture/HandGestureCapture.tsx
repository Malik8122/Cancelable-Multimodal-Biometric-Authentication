import { AnimatePresence, motion } from 'motion/react'
import { AlertTriangle, CameraOff, Check, Circle, Loader2, RotateCcw, Square } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { HandLandmarker } from '@mediapipe/tasks-vision'
import { GESTURE_LABEL, MAX_RECORDING_MS, MIN_TRACKED_FRAMES_HINT, Z_GUIDE_PATH, Z_GUIDE_POINTS } from '../../config/hand'
import { loadHandLandmarker, toPayload, type GestureCapturePayload, type GestureFrame } from '../../hand/landmarker'
import { rise } from '../../lib/motion'
import { trailPath } from '../../lib/biometricVisuals'
import { GestureTrajectory } from './GestureTrajectory'
import { CaptureCard } from '../biometric/CaptureCard'
import { CaptureStatus, type StatusTone } from '../biometric/CaptureStatus'
import { useDiagnosticsEnabled } from '../../hooks/useDiagnosticsEnabled'
import { DiagnosticsBox } from '../diagnostics/DiagnosticsBox'

/** What the parent reports after it submitted the capture (processing / backend verdict). */
export interface HandCaptureFeedback {
  kind: 'processing' | 'ok' | 'error'
  message: string
}

interface Props {
  mode: 'register' | 'verify'
  onCapture: (payload: GestureCapturePayload) => void
  /** e.g. "Sample 02 / 03" during enrollment. */
  attemptLabel?: string
  feedback?: HandCaptureFeedback | null
  disabled?: boolean
  /** Change to clear the current recording (e.g. after a rejected attempt). */
  resetKey?: number
}

type Phase = 'loading' | 'error' | 'live' | 'recording' | 'captured'

// Palm centre = wrist + four finger MCPs (the same reference the backend uses for the trajectory).
const PALM = [0, 5, 9, 13, 17]
const HAND_LOST_STOP_MS = 600
// React only re-renders the elapsed-time label this often; the progress bar is updated directly every frame.
const ELAPSED_UI_MS = 100

export function HandGestureCapture({ mode, onCapture, attemptLabel, feedback, disabled, resetKey }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const landmarkerRef = useRef<HandLandmarker | null>(null)
  const phaseRef = useRef<Phase>('loading')
  const framesRef = useRef<GestureFrame[]>([])
  const inferenceRef = useRef<number[]>([])
  const trailRef = useRef<[number, number][]>([])
  const startRef = useRef(0)
  const lastSeenRef = useRef(0)
  const lastVideoTimeRef = useRef(-1)
  const handVisibleRef = useRef(false)
  const lastElapsedUiRef = useRef(0)
  const progressRef = useRef<HTMLDivElement>(null)
  const [phase, setPhaseState] = useState<Phase>('loading')
  const [handVisible, setHandVisible] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [summary, setSummary] = useState<{ tracked: number; seconds: number } | null>(null)
  const [elapsed, setElapsed] = useState(0)
  // Replay of the captured palm trajectory (real tracked points): 'draw' over the camera, then 'thumb' in the corner.
  const [replay, setReplay] = useState<{ path: string; phase: 'draw' | 'thumb' } | null>(null)

  const setPhase = (next: Phase) => {
    phaseRef.current = next
    setPhaseState(next)
  }

  const stopRecording = useCallback(() => {
    if (phaseRef.current !== 'recording') return
    const video = videoRef.current
    const frames = framesRef.current
    const tracked = frames.filter((f) => f.landmarks).length
    const seconds = frames.length > 1 ? (frames[frames.length - 1].t - frames[0].t) / 1000 : 0
    setSummary({ tracked, seconds })
    const path = trailPath(trailRef.current, (video?.videoWidth ?? 640) / (video?.videoHeight ?? 480))
    setReplay(path ? { path, phase: 'draw' } : null)
    const canvas = canvasRef.current
    canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
    setPhase('captured')
    onCapture(toPayload(frames, video?.videoWidth ?? 640, video?.videoHeight ?? 480, inferenceRef.current))
  }, [onCapture])

  // Camera + model, once.
  useEffect(() => {
    let cancelled = false
    navigator.mediaDevices
      ?.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } })
      .then(async (stream) => {
        if (cancelled) return stream.getTracks().forEach((track) => track.stop())
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          void videoRef.current.play().catch(() => {})
        }
        landmarkerRef.current = await loadHandLandmarker()
        if (!cancelled) setPhase('live')
      })
      .catch((err) => {
        if (cancelled) return
        setError(err instanceof Error ? err.message : 'Camera or hand tracker could not be started.')
        setPhase('error')
      })
    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])

  // Per-frame hand tracking (only on new video frames).
  useEffect(() => {
    let raf = 0
    const tick = () => {
      raf = requestAnimationFrame(tick)
      const video = videoRef.current
      const landmarker = landmarkerRef.current
      const current = phaseRef.current
      if (!video || !landmarker || video.readyState < 2 || (current !== 'live' && current !== 'recording')) return
      if (video.currentTime === lastVideoTimeRef.current) return
      lastVideoTimeRef.current = video.currentTime
      const now = performance.now()
      const result = landmarker.detectForVideo(video, now)
      const took = performance.now() - now
      const hand = result.landmarks[0]
      const points = hand ? hand.map((p) => [round(p.x), round(p.y), round(p.z)]) : null
      if (hand) lastSeenRef.current = now
      if (handVisibleRef.current !== !!hand) {
        handVisibleRef.current = !!hand
        setHandVisible(!!hand)
      }
      if (current === 'recording') {
        const t = now - startRef.current
        framesRef.current.push({ t: round(t, 1), landmarks: points, handedness: result.handedness[0]?.[0]?.categoryName })
        inferenceRef.current.push(took)
        if (hand) {
          const cx = PALM.reduce((s, i) => s + hand[i].x, 0) / PALM.length
          const cy = PALM.reduce((s, i) => s + hand[i].y, 0) / PALM.length
          trailRef.current.push([cx, cy])
        }
        if (progressRef.current) progressRef.current.style.transform = `scaleX(${Math.min(1, t / MAX_RECORDING_MS)})`
        if (t - lastElapsedUiRef.current >= ELAPSED_UI_MS) {
          lastElapsedUiRef.current = t
          setElapsed(t)
        }
        const tracked = trailRef.current.length
        if (t >= MAX_RECORDING_MS || (tracked >= MIN_TRACKED_FRAMES_HINT && now - lastSeenRef.current > HAND_LOST_STOP_MS)) {
          stopRecording()
        }
      }
      draw(canvasRef.current, video, hand ?? null, trailRef.current)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [stopRecording])

  const reset = useCallback(() => {
    framesRef.current = []
    inferenceRef.current = []
    trailRef.current = []
    setSummary(null)
    setReplay(null)
    setElapsed(0)
    lastElapsedUiRef.current = 0
    if (phaseRef.current === 'captured') setPhase('live')
  }, [])

  useEffect(() => {
    if (resetKey !== undefined) reset()
  }, [resetKey, reset])

  const start = () => {
    reset()
    startRef.current = performance.now()
    lastSeenRef.current = performance.now()
    setPhase('recording')
  }

  const status: { tone: StatusTone; label: string } =
    phase === 'loading'
      ? { tone: 'processing', label: 'Starting camera and hand tracker...' }
      : phase === 'error'
        ? { tone: 'danger', label: 'Camera unavailable' }
        : phase === 'recording'
          ? handVisible
            ? { tone: 'live', label: 'Tracking trajectory' }
            : { tone: 'warning', label: 'Hand lost - keep it in view' }
          : phase === 'captured'
            ? feedback?.kind === 'processing'
              ? { tone: 'processing', label: 'Analyzing movement...' }
              : feedback?.kind === 'error'
                ? { tone: 'warning', label: mode === 'register' ? 'Sample not accepted' : 'Gesture not accepted' }
                : feedback?.kind === 'ok' && mode === 'register'
                  ? { tone: 'success', label: 'Sample captured' }
                  : { tone: 'success', label: summary ? `${mode === 'register' ? 'Sample' : 'Gesture'} captured - ${summary.seconds.toFixed(1)} s, ${summary.tracked} frames` : 'Gesture captured' }
            : handVisible
              ? { tone: 'live', label: 'Hand detected' }
              : { tone: 'idle', label: 'Waiting for hand...' }

  const showGuide = phase === 'live' || phase === 'recording'
  const diagnostics = useDiagnosticsEnabled()

  return (
    <CaptureCard
      modality="hand"
      eyebrow={mode === 'register' ? 'Dynamic hand gesture' : 'Dynamic hand verification'}
      title="Z gesture"
      state={phase === 'recording' ? 'active' : phase === 'captured' && feedback?.kind !== 'error' ? 'complete' : phase === 'error' || feedback?.kind === 'error' ? 'attention' : 'idle'}
      aside={
        attemptLabel ? (
          <span className="rounded-full border border-border-strong bg-raised/70 px-2.5 py-1 text-xs font-medium tabular-nums text-foreground">{attemptLabel}</span>
        ) : undefined
      }
    >
      <div className="relative mb-3 aspect-video overflow-hidden rounded-xl border border-border bg-black/50">
        <video ref={videoRef} autoPlay muted playsInline className="h-full w-full -scale-x-100 object-cover" />
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100" />

        {/* What to draw: a faint Z guide (start dot highlighted). While waiting it traces itself once every few seconds;
            once the hand is being tracked it stays still, so it never competes with the user's own trajectory. It is a
            guide only - never presented as the user's gesture. */}
        {/* Capture boundary: where the gesture should happen. */}
        {showGuide && <div aria-hidden className="pointer-events-none absolute inset-[9%] rounded-2xl border border-dashed border-white/15" />}
        {showGuide && (
          <svg aria-hidden viewBox="0 0 200 100" className="pointer-events-none absolute inset-0 m-auto h-[46%] w-[62%] opacity-70">
            <path d={Z_GUIDE_PATH} fill="none" strokeWidth={1.5} strokeLinejoin="round" className="stroke-white/15" strokeDasharray="3 5" />
            {Z_GUIDE_POINTS.map(([x, y], i) => (
              <circle key={i} cx={x} cy={y} r={i === 0 ? 3.5 : 2.2} className={i === 0 ? 'fill-info/70' : 'fill-white/25'} />
            ))}
            {phase === 'live' && !handVisible && (
              <motion.path
                d={Z_GUIDE_PATH}
                strokeLinejoin="round"
                fill="none"
                strokeWidth={2}
                strokeLinecap="round"
                className="stroke-info/70"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: [0, 1, 1], opacity: [0, 1, 0] }}
                transition={{ duration: 3.2, repeat: Infinity, ease: 'easeInOut', times: [0, 0.75, 1] }}
              />
            )}
          </svg>
        )}

        {phase === 'error' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/70 px-6 text-center">
            <CameraOff className="h-6 w-6 text-danger" strokeWidth={1.5} aria-hidden />
            <p className="text-sm font-medium text-foreground">Camera unavailable</p>
            <p className="max-w-xs text-xs text-muted-foreground">Check camera permissions and reload the page. {error}</p>
          </div>
        )}

        {/* Replay: the real palm trajectory draws itself, then shrinks into a thumbnail of this capture. */}
        <AnimatePresence>
          {replay && phase === 'captured' && (
            <motion.div
              key="replay"
              layout
              transition={{ type: 'spring', stiffness: 260, damping: 30 }}
              className={
                replay.phase === 'draw'
                  ? 'pointer-events-none absolute inset-[12%] z-10'
                  : 'pointer-events-none absolute right-3 bottom-3 z-10 h-16 w-16 rounded-xl border border-success/40 bg-black/70 p-1.5'
              }
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <GestureTrajectory
                path={replay.path}
                draw
                strokeWidth={replay.phase === 'draw' ? 3 : 5}
                label="Replay of the captured hand trajectory"
                onDrawn={() => setReplay((r) => (r ? { ...r, phase: 'thumb' } : r))}
              />
            </motion.div>
          )}
        </AnimatePresence>

        <div className="absolute left-3 top-3 right-3 z-20 flex items-center justify-between gap-2">
          <CaptureStatus overlay tone={status.tone} label={status.label} />
          {phase === 'recording' && (
            <span className="rounded-full bg-black/65 px-2 py-0.5 font-mono text-xs tabular-nums text-white/90">{(elapsed / 1000).toFixed(1)} s</span>
          )}
        </div>

        {phase === 'recording' && (
          <div className="absolute inset-x-0 bottom-0 h-1 bg-white/10" aria-hidden>
            <div ref={progressRef} className="h-full origin-left bg-info" style={{ transform: 'scaleX(0)' }} />
          </div>
        )}
      </div>

      {/* Development only: what the browser actually tracked for this sample (before the backend's checks). */}
      {diagnostics && phase === 'captured' && summary && (
        <DiagnosticsBox
          title="Client tracking (this sample)"
          className="mb-3"
          rows={[
            { label: 'Frames with a hand', value: String(summary.tracked) },
            { label: 'Recorded duration', value: `${summary.seconds.toFixed(2)} s` },
            {
              label: 'Tracking rate >= 10 fps',
              value: `${summary.seconds > 0 ? (summary.tracked / summary.seconds).toFixed(1) : '-'} fps`,
              status: summary.seconds > 0 && summary.tracked / summary.seconds >= 10 ? 'pass' : 'fail',
            },
          ]}
        />
      )}
      <AnimatePresence initial={false} mode="popLayout">
        {feedback && (
          <motion.p
            key={`${feedback.kind}:${feedback.message}`}
            variants={rise}
            initial="hidden"
            animate="show"
            exit="exit"
            role={feedback.kind === 'error' ? 'alert' : undefined}
            className={`mb-3 flex items-start gap-2 rounded-lg border px-3 py-2 text-xs ${
              feedback.kind === 'ok'
                ? 'border-success/30 bg-success/10 text-success'
                : feedback.kind === 'error'
                  ? 'border-warning/35 bg-warning/10 text-warning'
                  : 'border-border text-muted-foreground'
            }`}
          >
            {feedback.kind === 'ok' ? (
              <Check className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={2.25} aria-hidden />
            ) : feedback.kind === 'error' ? (
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={2} aria-hidden />
            ) : (
              <Loader2 className="mt-0.5 h-3.5 w-3.5 shrink-0 animate-spin" strokeWidth={2} aria-hidden />
            )}
            {feedback.message}
          </motion.p>
        )}
      </AnimatePresence>

      <p className="mb-4 text-[13px] leading-relaxed text-muted-foreground">
        {mode === 'register'
          ? `This sample: perform the ${GESTURE_LABEL} naturally - press Record, then draw one continuous Z in the air with one hand (left to right, diagonally down to the left, then left to right) and lower your hand or press Stop. Each sample is its own performance; it does not need to copy the previous one.`
          : 'Perform the Z gesture once for verification: press Record, draw the Z as you enrolled it (left to right, diagonally down to the left, then left to right), then lower your hand or press Stop.'}
      </p>

      <div className="flex gap-2.5">
        {phase === 'recording' ? (
          <button type="button" onClick={stopRecording} className="btn btn-danger flex-1">
            <Square className="h-4 w-4" strokeWidth={1.5} aria-hidden />
            Stop
          </button>
        ) : phase === 'captured' ? (
          <button type="button" onClick={reset} disabled={disabled} className="btn btn-secondary flex-1">
            <RotateCcw className="h-4 w-4" strokeWidth={1.5} aria-hidden />
            Record again
          </button>
        ) : (
          <motion.button type="button" whileTap={{ scale: 0.98 }} onClick={start} disabled={disabled || phase !== 'live'} className="btn btn-primary flex-1">
            <Circle className="h-4 w-4" strokeWidth={1.5} aria-hidden />
            Record gesture
          </motion.button>
        )}
      </div>
      <p className="mt-3 text-center text-meta text-muted-foreground">
        Hand tracking runs in your browser. Only landmark coordinates are sent - never video.
      </p>
    </CaptureCard>
  )
}

function round(value: number, digits = 5): number {
  const f = 10 ** digits
  return Math.round(value * f) / f
}

// Live overlay (never stored): the palm-centre trajectory of the current recording, fading from old to new, plus a
// small reticle on the palm and faint landmarks - enough to show "the hand is being tracked" without a busy skeleton.
function draw(canvas: HTMLCanvasElement | null, video: HTMLVideoElement, hand: { x: number; y: number }[] | null, trail: [number, number][]) {
  if (!canvas) return
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
  }
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const w = canvas.width
  const h = canvas.height
  ctx.clearRect(0, 0, w, h)
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  if (trail.length > 1) {
    for (let i = 1; i < trail.length; i++) {
      const age = i / trail.length
      ctx.strokeStyle = `rgba(106, 147, 240, ${0.15 + 0.8 * age})`
      ctx.lineWidth = 2 + 2.5 * age
      ctx.beginPath()
      ctx.moveTo(trail[i - 1][0] * w, trail[i - 1][1] * h)
      ctx.lineTo(trail[i][0] * w, trail[i][1] * h)
      ctx.stroke()
    }
  }
  if (hand) {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.35)'
    for (const p of hand) {
      ctx.beginPath()
      ctx.arc(p.x * w, p.y * h, 2, 0, Math.PI * 2)
      ctx.fill()
    }
    const cx = (PALM.reduce((s, i) => s + hand[i].x, 0) / PALM.length) * w
    const cy = (PALM.reduce((s, i) => s + hand[i].y, 0) / PALM.length) * h
    ctx.strokeStyle = 'rgba(76, 195, 224, 0.95)'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.arc(cx, cy, 10, 0, Math.PI * 2)
    ctx.stroke()
    ctx.fillStyle = 'rgba(76, 195, 224, 0.95)'
    ctx.beginPath()
    ctx.arc(cx, cy, 3, 0, Math.PI * 2)
    ctx.fill()
  }
}
