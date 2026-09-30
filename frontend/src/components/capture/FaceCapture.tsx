import { motion, AnimatePresence } from 'motion/react'
import { Camera, CameraOff, RotateCcw, Upload } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useReducedMotion } from '../../hooks/useReducedMotion'
import { CaptureCard } from '../biometric/CaptureCard'
import { CaptureStatus, type StatusTone } from '../biometric/CaptureStatus'

interface Props {
  mode: 'register' | 'verify'
  onCapture: (blob: Blob, filename: string) => void
  disabled?: boolean
}

const ACCEPTED = ['image/jpeg', 'image/jpg', 'image/png']

// Corner brackets of the face guide and how far each snaps inward on lock-on.
const CORNERS = [
  { cls: '-top-0.5 -left-0.5 border-t-2 border-l-2 rounded-tl-lg', dx: 10, dy: 10 },
  { cls: '-top-0.5 -right-0.5 border-t-2 border-r-2 rounded-tr-lg', dx: -10, dy: 10 },
  { cls: '-bottom-0.5 -left-0.5 border-b-2 border-l-2 rounded-bl-lg', dx: 10, dy: -10 },
  { cls: '-bottom-0.5 -right-0.5 border-b-2 border-r-2 rounded-br-lg', dx: -10, dy: -10 },
]

export function FaceCapture({ mode, onCapture, disabled }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [streamActive, setStreamActive] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [flash, setFlash] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const reducedMotion = useReducedMotion()

  // <video> below is now ALWAYS mounted (never conditionally rendered), so
  // videoRef.current already exists by the time this effect's promise
  // resolves - srcObject can be assigned directly here, with no second
  // effect needed to wait for a later render. A previous fix moved the
  // assignment into a second useEffect keyed on streamActive instead of
  // fixing the actual issue (the element not existing yet) - that only
  // worked if the video element mounting actually happened synchronously
  // with that state flip, which conditional rendering doesn't guarantee
  // robustly across browsers. Explicit .play() is called defensively after
  // assigning srcObject: `autoplay` should suffice per spec once srcObject
  // is set on an already-mounted element, but a handful of Chromium/WebView
  // builds don't reliably resume it for a src assigned after initial mount.
  useEffect(() => {
    let cancelled = false
    navigator.mediaDevices
      ?.getUserMedia({ video: { facingMode: 'user' } })
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((track) => track.stop())
          return
        }
        streamRef.current = s
        if (videoRef.current) {
          videoRef.current.srcObject = s
          void videoRef.current.play().catch(() => {})
        }
        setStreamActive(true)
      })
      .catch((err) => setCameraError(err instanceof Error ? err.message : 'Camera access was denied.'))

    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])

  const capture = () => {
    const video = videoRef.current
    if (!video) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    ctx?.drawImage(video, 0, 0)
    canvas.toBlob((blob) => {
      if (blob) {
        onCapture(blob, 'face.png')
        setPreviewUrl(canvas.toDataURL('image/png'))
      }
    }, 'image/png')
    setFlash(true)
    window.setTimeout(() => setFlash(false), 220)
  }

  const retake = () => setPreviewUrl(null)

  const handleFile = (file: File) => {
    onCapture(file, file.name)
    setPreviewUrl(URL.createObjectURL(file))
  }

  const phase = cameraError && !previewUrl ? 'error' : previewUrl ? 'captured' : streamActive ? 'live' : 'starting'
  const status: { tone: StatusTone; label: string } =
    phase === 'error'
      ? { tone: 'danger', label: 'Camera unavailable' }
      : phase === 'captured'
        ? { tone: 'success', label: 'Face captured' }
        : phase === 'live'
          ? { tone: 'live', label: 'Camera live - align your face' }
          : { tone: 'processing', label: 'Initializing camera...' }

  return (
    <CaptureCard
      modality="face"
      eyebrow={mode === 'register' ? 'Face enrollment' : 'Face verification'}
      title={mode === 'register' ? 'Capture your face' : 'Look at the camera'}
      state={phase === 'captured' ? 'complete' : phase === 'error' ? 'attention' : 'idle'}
    >
      <div className="relative mb-4 aspect-video overflow-hidden rounded-xl border border-border bg-black/50">
        <video ref={videoRef} autoPlay muted playsInline className="h-full w-full object-cover" />

        {streamActive && !previewUrl && (
          <>
            <AnimatePresence>
              {flash && (
                <motion.div
                  initial={{ opacity: 0.7 }}
                  animate={{ opacity: 0 }}
                  exit={{ opacity: 0 }}
                  className="pointer-events-none absolute inset-0 bg-white"
                />
              )}
            </AnimatePresence>
          </>
        )}

        {/* Face guide over the live camera AND the captured frame: the brackets frame the face while live (with one slow
            scan line) and lock on - snap inward, pulse once, turn green - when the capture is taken. Detection itself
            happens on the backend at submit, so the UI never claims a face was detected before that. */}
        {(streamActive || previewUrl) && (
          <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
            <div className="relative h-[72%] w-[46%] min-w-32">
              {CORNERS.map((corner) => (
                <motion.div
                  key={corner.cls}
                  className={`absolute h-6 w-6 ${corner.cls}`}
                  initial={false}
                  animate={{
                    x: previewUrl ? corner.dx : 0,
                    y: previewUrl ? corner.dy : 0,
                    borderColor: previewUrl ? 'var(--color-success)' : 'color-mix(in srgb, var(--color-info) 80%, transparent)',
                  }}
                  transition={{ type: 'spring', stiffness: 420, damping: 22 }}
                />
              ))}
              <AnimatePresence>
                {previewUrl && (
                  <motion.div
                    key="lock-pulse"
                    className="absolute inset-3 rounded-2xl border-2 border-success"
                    initial={{ opacity: 0.8, scale: 0.94 }}
                    animate={{ opacity: 0, scale: 1.06 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.7, ease: 'easeOut', delay: 0.12 }}
                  />
                )}
              </AnimatePresence>
              {streamActive && !previewUrl && !reducedMotion && (
                <motion.div
                  className="absolute inset-x-2 h-px bg-gradient-to-r from-transparent via-info/70 to-transparent"
                  animate={{ top: ['6%', '94%', '6%'] }}
                  transition={{ duration: 3.6, repeat: Infinity, ease: 'easeInOut' }}
                />
              )}
            </div>
          </div>
        )}

        <AnimatePresence>
          {previewUrl && (
            <motion.img
              key="preview"
              src={previewUrl}
              alt="Captured face"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 h-full w-full object-cover"
            />
          )}
        </AnimatePresence>

        {phase === 'error' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/70 px-6 text-center">
            <CameraOff className="h-6 w-6 text-danger" strokeWidth={1.5} aria-hidden />
            <p className="text-sm font-medium text-foreground">Camera unavailable</p>
            <p className="max-w-xs text-xs text-muted-foreground">Check camera permissions and try again, or upload a photo instead.</p>
          </div>
        )}

        <div className="absolute left-3 top-3 z-20">
          <CaptureStatus overlay tone={status.tone} label={status.label} />
        </div>
      </div>

      <p className="mb-4 text-[13px] leading-relaxed text-muted-foreground">
        Face the camera inside the frame, with even lighting. Remove glasses if possible.
      </p>

      <div className="flex gap-2.5">
        {previewUrl ? (
          <button type="button" onClick={retake} className="btn btn-secondary flex-1">
            <RotateCcw className="h-4 w-4" strokeWidth={1.5} aria-hidden />
            Retake
          </button>
        ) : (
          <motion.button type="button" whileTap={{ scale: 0.98 }} onClick={capture} disabled={disabled || !streamActive} className="btn btn-primary flex-1">
            <Camera className="h-4 w-4" strokeWidth={1.5} aria-hidden />
            Capture
          </motion.button>
        )}
        <label className="btn btn-secondary text-muted-foreground has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-ring">
          <Upload className="h-4 w-4" strokeWidth={1.5} aria-hidden />
          <span className="hidden sm:inline">Upload instead</span>
          <span className="sm:hidden">Upload</span>
          <input
            type="file"
            accept={ACCEPTED.join(',')}
            className="sr-only"
            disabled={disabled}
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
        </label>
      </div>
      <p className="mt-3 text-center text-meta text-muted-foreground">Accepted formats: PNG, JPG, JPEG</p>
    </CaptureCard>
  )
}
