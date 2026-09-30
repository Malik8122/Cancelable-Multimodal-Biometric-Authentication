import { AnimatePresence, motion } from 'motion/react'
import { AlertCircle, Camera, CameraOff, Check, Loader2, Upload } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { checkFacePose } from '../../api/client'
import { ApiError, type FacePose } from '../../api/types'
import { DURATION, EASE_OUT, rise } from '../../lib/motion'
import { CaptureStatus, type StatusTone } from '../biometric/CaptureStatus'

export const FACE_POSE_ORDER: FacePose[] = ['front', 'left', 'right', 'up', 'down']

// Five full-face, forward-facing captures, presented to the user as ONE "Face Registration"
// process (not five separate named steps) - NOT deliberate head turns, and no directional
// instructions, pose names, or arrows are shown anywhere in this component. The face
// preprocessing pipeline (preprocessing/face.py) crops by bounding box only, with no
// landmark-based rotation correction, so a deliberately rotated enrollment capture measurably
// hurts the resulting centroid's similarity to a normal, frontal live authentication capture
// (see the alignment investigation). The keys (front/left/right/up/down) below are unchanged
// internal identifiers only - the five-accepted-embeddings mechanism underneath (and everything
// backend/API-side) is unchanged; this is a presentation-only simplification.

const ACCEPTED = ['image/jpeg', 'image/jpg', 'image/png']

//: How long the "Sample captured" confirmation shows before the UI returns to the idle prompt.
const CAPTURED_MESSAGE_MS = 1100

interface Props {
  /** Called whenever the set of accepted poses changes. The enrollment is ready when all five are present. */
  onChange: (poses: Partial<Record<FacePose, Blob>>) => void
  disabled?: boolean
}

// One-time "Face Registration": a single camera preview with a simple sample-count progress
// indicator, backed by the existing five-accepted-capture mechanism (five embeddings -> one
// centroid, unchanged). Each capture is checked by the backend immediately (face detection +
// quality gating, nothing stored); a rejected capture - no face, more than one face, blurry, too
// small/off-center/angled in frame, or low detection confidence - is retaken on the spot, exactly
// as before. Authentication does NOT use this component: it is a single camera capture with no
// guided prompts.
export function GuidedFaceCapture({ onChange, disabled }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const cancelledRef = useRef(false)
  const capturedMessageTimeoutRef = useRef<number | null>(null)
  // Set synchronously on click, before the async frame grab: a second click (double-click, or a click while the
  // PNG is still encoding) can never start a parallel capture for the same slot.
  const inFlightRef = useRef(false)

  const [streamActive, setStreamActive] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [poses, setPoses] = useState<Partial<Record<FacePose, Blob>>>({})
  const [checking, setChecking] = useState(false)
  const [rejection, setRejection] = useState<string | null>(null)
  const [justCaptured, setJustCaptured] = useState(false)

  useEffect(() => {
    cancelledRef.current = false
    navigator.mediaDevices
      ?.getUserMedia({ video: { facingMode: 'user' } })
      .then((stream) => {
        if (cancelledRef.current) {
          stream.getTracks().forEach((track) => track.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          void videoRef.current.play().catch(() => {})
        }
        setStreamActive(true)
      })
      .catch((err) => setCameraError(err instanceof Error ? err.message : 'Camera access was denied.'))
    return () => {
      cancelledRef.current = true
      streamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])

  useEffect(() => {
    return () => {
      if (capturedMessageTimeoutRef.current !== null) window.clearTimeout(capturedMessageTimeoutRef.current)
    }
  }, [])

  // The progress indicator reflects only ACCEPTED captures - a rejected/retaken capture never
  // advances it, since `poses` only ever gains an entry once the backend has confirmed VALID.
  const acceptedCount = FACE_POSE_ORDER.filter((pose) => poses[pose]).length
  const complete = acceptedCount === FACE_POSE_ORDER.length
  // The next slot to fill is always the first EMPTY one, derived from the accepted captures themselves - never a
  // separate step counter, which could drift past an unfilled slot and leave the UI stuck at 4 of 5.
  const currentPose = FACE_POSE_ORDER.find((pose) => !poses[pose]) ?? FACE_POSE_ORDER[FACE_POSE_ORDER.length - 1]
  const currentIndex = FACE_POSE_ORDER.indexOf(currentPose)

  const grabFrame = (): Promise<Blob | null> => {
    const video = videoRef.current
    if (!video || !video.videoWidth) return Promise.resolve(null)
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d')?.drawImage(video, 0, 0)
    // PNG (lossless), matching FaceCapture.tsx's authentication capture exactly - previously this
    // used lossy JPEG (q=0.92) while authentication used PNG, an unnecessary encoding difference
    // between the two capture paths (see the alignment investigation).
    return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob), 'image/png'))
  }

  // Ask the backend whether this capture is usable; accept it and move on, or explain and let the user retake.
  const submitCapture = async (blob: Blob | null, filename: string, pose: FacePose) => {
    if (!blob) {
      setRejection('The camera did not return an image. Please try again.')
      return
    }
    setChecking(true)
    setRejection(null)
    try {
      const verdict = await checkFacePose(pose, blob, filename)
      if (!verdict.valid) {
        setRejection(verdict.detail ?? 'This capture cannot be used. Please try again.')
        return
      }
      setPoses((prev) => ({ ...prev, [pose]: blob }))
      setJustCaptured(true)
      if (capturedMessageTimeoutRef.current !== null) window.clearTimeout(capturedMessageTimeoutRef.current)
      capturedMessageTimeoutRef.current = window.setTimeout(() => setJustCaptured(false), CAPTURED_MESSAGE_MS)
    } catch (err) {
      setRejection(err instanceof ApiError ? err.detail : 'The backend is unreachable.')
    } finally {
      setChecking(false)
    }
  }

  // Report every change of the accepted set (after the functional update above has been applied).
  useEffect(() => {
    onChange(poses)
  }, [poses, onChange])

  // One capture at a time, always for the first empty slot; the guard is released only when the check has finished.
  const capture = (source: () => Promise<Blob | null>, filename: (pose: FacePose) => string) => {
    if (inFlightRef.current || complete) return
    inFlightRef.current = true
    const pose = currentPose
    void source()
      .then((blob) => submitCapture(blob, filename(pose), pose))
      .finally(() => {
        inFlightRef.current = false
      })
  }

  const status: { tone: StatusTone; label: string } = cameraError && !streamActive
    ? { tone: 'danger', label: 'Camera unavailable' }
    : !streamActive
      ? { tone: 'processing', label: 'Initializing camera...' }
      : complete
        ? { tone: 'success', label: 'Face registration complete' }
        : checking
          ? { tone: 'processing', label: 'Checking sample...' }
          : rejection
            ? { tone: 'warning', label: 'Sample not usable - retake' }
            : justCaptured
              ? { tone: 'success', label: `Sample ${acceptedCount} accepted` }
              : { tone: 'live', label: 'Camera live - hold still' }

  // Progress ring around the camera: one fifth per ACCEPTED sample.
  const ringColor = rejection ? 'stroke-warning' : 'stroke-success'

  return (
    <div className="rounded-xl border border-border bg-raised/30 p-4 sm:p-5">
      {/* "Face Registration" is this section's header (the enclosing EnrollmentCard's title in
          RegisterPage.tsx) - this is its subtitle, not a second header. */}
      <p className="mb-4 text-center text-sm text-muted-foreground">
        Look naturally at the camera. We&apos;ll take a few quick samples to create your face profile.
      </p>

      {/* Circular face guide with a progress ring */}
      <div className="mx-auto mb-4 flex justify-center">
        <div className="relative h-60 w-60 sm:h-64 sm:w-64">
          <svg aria-hidden viewBox="0 0 100 100" className="absolute -inset-2 h-[calc(100%+1rem)] w-[calc(100%+1rem)] -rotate-90">
            <circle cx="50" cy="50" r="48" fill="none" strokeWidth="1.5" className="stroke-border-strong" />
            <motion.circle
              cx="50"
              cy="50"
              r="48"
              fill="none"
              strokeWidth="2"
              strokeLinecap="round"
              className={ringColor}
              initial={false}
              animate={{ pathLength: acceptedCount / FACE_POSE_ORDER.length }}
              transition={{ duration: DURATION.slow, ease: EASE_OUT }}
            />
          </svg>
          <div className="absolute inset-0 overflow-hidden rounded-full border border-border-strong bg-black/50">
            <video ref={videoRef} autoPlay muted playsInline className="h-full w-full scale-x-[-1] object-cover" />
            {!streamActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 px-8 text-center">
                {cameraError ? (
                  <>
                    <CameraOff className="h-6 w-6 text-danger" strokeWidth={1.5} aria-hidden />
                    <p className="text-xs text-muted-foreground">Check camera permissions and try again, or upload photos instead.</p>
                  </>
                ) : (
                  <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" strokeWidth={1.5} aria-hidden />
                )}
              </div>
            )}
          </div>
          <AnimatePresence>
            {complete && (
              <motion.span
                initial={{ scale: 0.5, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: DURATION.base, ease: EASE_OUT }}
                className="absolute right-3 bottom-3 flex h-9 w-9 items-center justify-center rounded-full bg-success text-success-foreground shadow-lg"
              >
                <Check className="h-5 w-5" strokeWidth={2.5} aria-hidden />
              </motion.span>
            )}
          </AnimatePresence>
        </div>
      </div>

      <div className="mb-4 flex flex-col items-center gap-2">
        <CaptureStatus tone={status.tone} label={status.label} />
        {/* Progress: accepted-sample count only - a rejected/retaken capture never advances this. */}
        <p className="text-xs tabular-nums text-muted-foreground">
          {acceptedCount} / {FACE_POSE_ORDER.length} samples accepted
        </p>
        <AnimatePresence initial={false}>
          {rejection && (
            <motion.p
              key={rejection}
              variants={rise}
              initial="hidden"
              animate="show"
              exit="exit"
              role="alert"
              className="mx-auto flex max-w-sm items-start justify-center gap-1.5 text-center text-xs text-warning"
            >
              <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.5} aria-hidden /> {rejection}
            </motion.p>
          )}
        </AnimatePresence>
      </div>

      {!complete && (
        <div className="flex gap-2.5">
          <motion.button
            type="button"
            whileTap={{ scale: 0.98 }}
            onClick={() => capture(grabFrame, (pose) => `face-${pose}.png`)}
            disabled={disabled || checking || !streamActive}
            className="btn btn-primary flex-1"
          >
            {checking ? <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} aria-hidden /> : <Camera className="h-4 w-4" strokeWidth={1.5} aria-hidden />}
            {checking ? 'Checking...' : rejection ? 'Retake' : `Capture sample ${currentIndex + 1}`}
          </motion.button>
          <label className="btn btn-secondary text-muted-foreground has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-ring">
            <Upload className="h-4 w-4" strokeWidth={1.5} aria-hidden />
            <span className="hidden sm:inline">Upload photo</span>
            <span className="sm:hidden">Upload</span>
            <input
              type="file"
              accept={ACCEPTED.join(',')}
              className="sr-only"
              disabled={disabled || checking}
              onChange={(e) => {
                const file = e.target.files?.[0]
                e.target.value = ''
                if (file) capture(() => Promise.resolve(file), () => file.name)
              }}
            />
          </label>
        </div>
      )}
      <p className="mt-3 text-center text-meta text-muted-foreground">
        You only do this once. Good lighting helps; only blurry or undetected captures are rejected. Photo formats: PNG, JPG, JPEG.
      </p>
    </div>
  )
}
