import { motion } from 'motion/react'
import { Check } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { checkHandGesture } from '../../api/client'
import { HAND_ENROLLMENT_SAMPLES, sampleCounter } from '../../config/protocol'
import { payloadBlob, type GestureCapturePayload } from '../../hand/landmarker'
import { DURATION, EASE_OUT } from '../../lib/motion'
import { gestureTrajectoryPath } from '../../lib/biometricVisuals'
import { GestureTrajectory } from './GestureTrajectory'
import { HandGestureCapture, type HandCaptureFeedback } from './HandGestureCapture'
import type { HandGestureCheckResponse } from '../../api/types'
import { useDiagnosticsEnabled } from '../../hooks/useDiagnosticsEnabled'
import { DiagnosticsBox } from '../diagnostics/DiagnosticsBox'

interface Props {
  /** The three accepted, independent Z samples - in capture order - once all passed the check (null until then). */
  onChange: (attempts: Blob[] | null) => void
  resetKey?: number
}

/**
 * Enrollment from three INDEPENDENT Z gesture samples (Sample 01 / 03 ... 03 / 03): each is its own recording - a
 * separate, natural performance of the Z, so the samples capture how the user's movement naturally varies. Each sample
 * is checked by the backend's capture gate and Z check (POST /enroll/hand/check) as soon as it is recorded, so a poor
 * one is retaken on the spot; accepted samples are appended, never replaced. Nothing is stored until the parent submits
 * all three to POST /enroll/hand (which also refuses the same sequence twice, or a sample unlike both others).
 */
const REPLAY_MS = 1700
const TOTAL = HAND_ENROLLMENT_SAMPLES

export function HandGestureEnrollment({ onChange, resetKey }: Props) {
  const [accepted, setAccepted] = useState<Blob[]>([])
  // The accepted samples' palm trajectories, for the thumbnails (display only).
  const [paths, setPaths] = useState<(string | null)[]>([])
  const [feedback, setFeedback] = useState<HandCaptureFeedback | null>(null)
  const [captureReset, setCaptureReset] = useState(0)
  const [busy, setBusy] = useState(false)
  // The backend's capture-gate verdict for the last sample (development diagnostics only).
  const [lastCheck, setLastCheck] = useState<(HandGestureCheckResponse & { sample: number }) | null>(null)
  const diagnostics = useDiagnosticsEnabled()

  useEffect(() => {
    setAccepted([])
    setPaths([])
    setFeedback(null)
    setCaptureReset((k) => k + 1)
    onChange(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resetKey])

  const total = TOTAL
  const done = accepted.length >= total
  const current = Math.min(accepted.length + 1, total)

  const handleCapture = useCallback(
    async (payload: GestureCapturePayload) => {
      const attempt = accepted.length + 1
      const blob = payloadBlob(payload)
      const label = sampleCounter(attempt, TOTAL)
      setBusy(true)
      setFeedback({ kind: 'processing', message: `Analyzing movement - sample ${label}...` })
      try {
        const verdict = await checkHandGesture(blob, attempt)
        setLastCheck({ ...verdict, sample: attempt })
        if (!verdict.passed) {
          setFeedback({ kind: 'error', message: `${verdict.message} - please record sample ${label} again.` })
          setCaptureReset((k) => k + 1)
          return
        }
        const next = [...accepted, blob]
        setAccepted(next)
        setPaths((prev) => [...prev, gestureTrajectoryPath(payload)])
        const complete = next.length >= TOTAL
        setFeedback({
          kind: 'ok',
          message: complete
            ? `All ${TOTAL} samples captured.`
            : `Sample ${label} captured. Ready for next sample - sample ${sampleCounter(attempt + 1, TOTAL)}: perform the gesture again, naturally.`,
        })
        onChange(complete ? next : null)
        // Let the capture's trajectory replay finish before the recorder resets for the next sample.
        if (!complete) window.setTimeout(() => setCaptureReset((k) => k + 1), REPLAY_MS)
      } catch (error) {
        setFeedback({ kind: 'error', message: error instanceof Error ? error.message : 'The gesture could not be checked.' })
        setCaptureReset((k) => k + 1)
      } finally {
        setBusy(false)
      }
    },
    [accepted, onChange],
  )

  return (
    <div>
      {/* The collection: one tile per enrollment sample, each showing that sample's real captured trajectory. */}
      <div className="mb-4">
        <p className="mb-3 text-[13px] leading-relaxed text-muted-foreground">
          Capture {total} independent Z gesture samples to learn your natural movement pattern. Each sample is a
          separate, natural performance of the Z - they do not need to match each other exactly.
        </p>
        <div className="mb-2 flex items-center justify-between text-xs">
          <span className="text-muted-foreground">Gesture samples</span>
          <span className={`flex items-center gap-1.5 font-medium tabular-nums ${done ? 'text-success' : 'text-foreground'}`}>
            {done && <Check className="h-3.5 w-3.5" strokeWidth={2.25} aria-hidden />}
            {done ? `All ${total} samples captured` : `${accepted.length} / ${total} captured`}
          </span>
        </div>
        <ol className="mx-auto grid max-w-xs grid-cols-3 gap-2 sm:gap-3" aria-label="Gesture samples">
          {Array.from({ length: total }, (_, i) => {
            const isDone = i < accepted.length
            const isCurrent = i === accepted.length && !done
            const path = paths[i]
            return (
              <li
                key={i}
                className={`relative flex aspect-square flex-col items-center justify-center rounded-xl border transition-colors duration-300 ${
                  isDone ? 'border-success/40 bg-success/[0.07]' : isCurrent ? 'border-primary/60 bg-primary/10' : 'border-border bg-raised/40'
                }`}
              >
                {isDone && path ? (
                  <motion.div className="h-full w-full p-2" initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: DURATION.slow, ease: EASE_OUT }}>
                    <GestureTrajectory path={path} draw strokeWidth={6} label={`Sample ${i + 1} trajectory`} />
                  </motion.div>
                ) : (
                  <span className={`text-sm font-medium tabular-nums ${isCurrent ? 'text-primary' : 'text-muted-foreground'}`}>{String(i + 1).padStart(2, '0')}</span>
                )}
                {isDone && (
                  <span className="absolute -top-1.5 -right-1.5 flex h-4 w-4 items-center justify-center rounded-full bg-success text-success-foreground">
                    <Check className="h-2.5 w-2.5" strokeWidth={3} aria-hidden />
                  </span>
                )}
                <span className="sr-only">
                  Sample {i + 1}: {isDone ? 'captured' : isCurrent ? 'current' : 'not recorded'}
                </span>
              </li>
            )
          })}
        </ol>
      </div>
      {diagnostics && lastCheck && (
        <DiagnosticsBox
          title={`Hand sample ${String(lastCheck.sample).padStart(2, '0')} diagnostics`}
          className="mb-3"
          rows={[
            { label: 'Frames detected / recorded', value: `${lastCheck.metrics.frames_detected ?? '-'} / ${lastCheck.metrics.frames_total ?? '-'}` },
            { label: 'Detection ratio', value: lastCheck.metrics.detection_ratio?.toFixed(2) ?? '-' },
            {
              label: 'Tracking rate >= 10 fps',
              value: lastCheck.metrics.tracking_fps !== undefined ? `${lastCheck.metrics.tracking_fps} fps` : '-',
              status: lastCheck.metrics.tracking_fps === undefined ? undefined : lastCheck.metrics.tracking_fps >= 10 ? 'pass' : 'fail',
            },
            { label: 'Recorded / movement duration', value: `${lastCheck.metrics.duration_s ?? '-'} s / ${lastCheck.metrics.motion_duration_s ?? '-'} s` },
            { label: 'Idle hand trimmed', value: lastCheck.metrics.idle_trimmed_s !== undefined ? `${lastCheck.metrics.idle_trimmed_s} s` : '-' },
            {
              label: 'Trajectory extent >= 1 palm',
              value: lastCheck.metrics.trajectory_extent_palms?.toFixed(2) ?? '-',
              status: lastCheck.metrics.trajectory_extent_palms === undefined ? undefined : lastCheck.metrics.trajectory_extent_palms >= 1 ? 'pass' : 'fail',
            },
            {
              label: 'Z validity',
              value: lastCheck.passed ? `PASS (strokes explain ${lastCheck.metrics.z_explained?.toFixed(2) ?? '-'} of the path)` : lastCheck.verdict === 'INVALID_TRAJECTORY' ? 'FAIL' : '-',
              status: lastCheck.passed ? 'pass' : lastCheck.verdict === 'INVALID_TRAJECTORY' ? 'fail' : undefined,
            },
            { label: 'Trajectory points x features', value: `${lastCheck.metrics.resampled_length ?? '-'} x ${lastCheck.metrics.feature_dim ?? '-'}` },
          ]}
          footer={
            <>
              Capture gate: <span className={lastCheck.passed ? 'text-success' : 'text-danger'}>{lastCheck.passed ? 'PASS' : 'FAIL'}</span> ({lastCheck.verdict})
              {!lastCheck.passed && ` - ${lastCheck.message}`}
            </>
          }
        />
      )}
      <HandGestureCapture
        mode="register"
        attemptLabel={done ? `All ${total} samples` : `Sample ${sampleCounter(current, total)}`}
        onCapture={handleCapture}
        feedback={feedback}
        disabled={busy || done}
        resetKey={captureReset}
      />
    </div>
  )
}
