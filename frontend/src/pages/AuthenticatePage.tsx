import { AnimatePresence, LayoutGroup, motion } from 'motion/react'
import { AlertCircle, ArrowLeft, ArrowRight, Check, RotateCcw, ShieldCheck, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import { authenticateFusion } from '../api/client'
import { ApiError, type Modality } from '../api/types'
import { FaceCapture } from '../components/capture/FaceCapture'
import { GestureTrajectory } from '../components/capture/GestureTrajectory'
import { HandGestureCapture } from '../components/capture/HandGestureCapture'
import { Voiceprint } from '../components/capture/Voiceprint'
import { VoiceSentencesCapture, type VoiceRecording, type VoiceSentences } from '../components/capture/VoiceSentencesCapture'
import { emptyVoiceSlots } from '../config/protocol'
import { ClearanceBadge } from '../components/biometric/ClearanceBadge'
import { EnrollmentStatusPill } from '../components/biometric/EnrollmentStatusPill'
import { FactorCard } from '../components/biometric/FactorCard'
import { FusionMeter, type FusionOutcome } from '../components/biometric/FusionMeter'
import { ModalityBadge } from '../components/biometric/ModalityBadge'
import { ProgressStepper, type Step } from '../components/biometric/ProgressStepper'
import { VerificationPipeline } from '../components/biometric/VerificationPipeline'
import { APPLICATION_ID } from '../config/app'
import { FACTORS, MODALITY_LABEL } from '../config/buildings'
import { useAmbient } from '../hooks/useAmbient'
import { useBuildings } from '../context/BuildingsContext'
import { useSession } from '../context/SessionContext'
import { useEnrollmentProfile } from '../hooks/useEnrollmentProfile'
import { useReducedMotion } from '../hooks/useReducedMotion'
import { loadHandLandmarker, payloadBlob } from '../hand/landmarker'
import { buildFusionSamples } from '../lib/authSamples'
import { gestureTrajectoryPath, voiceprintLevels } from '../lib/biometricVisuals'
import { rise } from '../lib/motion'

type Phase = 'select' | 'capture' | 'processing' | 'error'

interface Captured {
  blob: Blob
  filename: string
}

// What presenting each factor involves in this session - the AUTHENTICATION protocol: one sample each.
const FACTOR_DETAIL: Record<Modality, string> = {
  face: 'One camera capture',
  voice: 'One spoken sentence',
  hand: 'One gesture sample',
}

/** Set by ResultPage's Retry after a RETRY_REQUESTED decision: the same factors, and the ones to capture again. */
interface RetryState {
  factors: Modality[]
  modalities: Modality[]
}

// The USER chooses which enrolled factors to present in this session; the building is only the context. The backend
// authenticates and fuses exactly the factors submitted. A factor that is not enrolled cannot be selected (and if one
// is submitted anyway the backend answers ENROLLMENT_REQUIRED, which is shown as its own screen).
export function AuthenticatePage() {
  const { buildingId } = useParams<{ buildingId: string }>()
  const navigate = useNavigate()
  const retry = (useLocation() as { state?: { retry?: RetryState } }).state?.retry
  const { userId, recordAttempt } = useSession()
  const { getBuilding, status: buildingsStatus } = useBuildings()
  const building = buildingId ? getBuilding(buildingId) : undefined
  const { enrolled, statuses, loading: profileLoading } = useEnrollmentProfile(userId)
  const reducedMotion = useReducedMotion()

  const [selected, setSelected] = useState<Modality[]>(() => retry?.factors ?? [])
  const [phase, setPhase] = useState<Phase>(() => (retry?.factors.length ? 'capture' : 'select'))
  const [captured, setCaptured] = useState<Partial<Record<Modality, Captured>>>({})
  // Voice authentication is two sentences (config/protocol.ts), each quality-checked before it is accepted.
  const [voiceSentences, setVoiceSentences] = useState<VoiceSentences>(() => emptyVoiceSlots<VoiceRecording>('verify'))
  // The capture deck: which factor is expanded into its capture experience, and which are being retaken.
  const [expanded, setExpanded] = useState<Modality | null>(null)
  const [retaking, setRetaking] = useState<Partial<Record<Modality, boolean>>>({})
  // A factor re-opened after it was captured shows its summary (with Retake); a factor captured while open keeps its
  // live capture view, so its capture moment (lock-on, voiceprint, trajectory replay) plays out.
  const [reviewing, setReviewing] = useState(false)
  // Display-only summaries of what was captured (real data): gesture trajectory and voiceprints.
  const [handPath, setHandPath] = useState<string | null>(null)
  const [voicePrint, setVoicePrint] = useState<number[] | null>(null)
  const [outcome, setOutcome] = useState<FusionOutcome>('pending')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [elapsedMs, setElapsedMs] = useState(0)
  const [showSlowNotice, setShowSlowNotice] = useState(false)
  const timerRef = useRef<number | null>(null)
  const startedAtRef = useRef(0)
  const abortControllerRef = useRef<AbortController | null>(null)

  useAmbient(
    phase === 'processing'
      ? outcome === 'granted'
        ? 'success'
        : outcome === 'denied'
          ? 'failure'
          : 'processing'
      : phase === 'error'
        ? 'failure'
        : phase === 'capture' && expanded
          ? 'capturing'
          : 'idle',
  )

  useEffect(() => {
    return () => abortControllerRef.current?.abort()
  }, [])

  // Warm up MediaPipe as soon as the hand gesture is chosen: the model download and GPU start-up then overlap with the
  // other captures, instead of starting only when the hand card is opened. (Cached - loaded once per page.)
  const handSelected = selected.includes('hand')
  useEffect(() => {
    if (handSelected) void loadHandLandmarker().catch(() => {})
  }, [handSelected])

  // Voiceprint of the first accepted sentence, for the voice card's summary.
  const voiceSentence = voiceSentences[0] ?? null
  useEffect(() => {
    let cancelled = false
    void (voiceSentence ? voiceprintLevels(voiceSentence.blob, 32) : Promise.resolve(null)).then((levels) => {
      if (!cancelled) setVoicePrint(levels)
    })
    return () => {
      cancelled = true
    }
  }, [voiceSentence])

  const faceBlob = captured.face?.blob
  const faceUrl = useMemo(() => (faceBlob ? URL.createObjectURL(faceBlob) : null), [faceBlob])
  useEffect(() => () => {
    if (faceUrl) URL.revokeObjectURL(faceUrl)
  }, [faceUrl])

  if (!building) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-16 text-center">
        <p className="text-muted-foreground">{buildingsStatus === 'loading' ? 'Loading facility...' : 'Unknown facility.'}</p>
        <Link to="/" className="mt-4 inline-block text-sm text-primary hover:underline">
          Return to the campus
        </Link>
      </div>
    )
  }

  const toggle = (modality: Modality) => {
    const isDeselecting = selected.includes(modality)
    setSelected((prev) => (isDeselecting ? prev.filter((m) => m !== modality) : [...prev, modality]))
    if (isDeselecting && modality === 'voice') setVoiceSentences(emptyVoiceSlots<VoiceRecording>('verify'))
    if (isDeselecting) {
      setCaptured((prev) => {
        if (!(modality in prev)) return prev
        const next = { ...prev }
        delete next[modality]
        return next
      })
    }
  }

  const isCaptured = (m: Modality) => (m === 'voice' ? voiceSentences.length > 0 && voiceSentences.every(Boolean) : !!captured[m])
  const allCaptured = selected.length > 0 && selected.every(isCaptured)
  const handleCapture = (modality: Modality) => (blob: Blob, filename: string) => {
    setCaptured((prev) => ({ ...prev, [modality]: { blob, filename } }))
  }

  const openFactor = (m: Modality) => {
    setReviewing(isCaptured(m))
    setExpanded(m)
  }
  const closeFactor = () => {
    setRetaking({})
    setExpanded(null)
  }
  // Continue: shrink this factor back into its card and open the next one that still needs a capture.
  const continueFrom = (m: Modality) => {
    const next = selected.find((other) => other !== m && !isCaptured(other))
    setRetaking({})
    setReviewing(false)
    setExpanded(next ?? null)
  }
  const retake = (m: Modality) => {
    setRetaking((prev) => ({ ...prev, [m]: true }))
    if (m === 'hand') setHandPath(null)
    if (m !== 'voice') {
      setCaptured((prev) => {
        const next = { ...prev }
        delete next[m]
        return next
      })
    }
  }

  const startTimer = () => {
    startedAtRef.current = performance.now()
    timerRef.current = window.setInterval(() => setElapsedMs(Math.round(performance.now() - startedAtRef.current)), 47)
  }
  const stopTimer = () => {
    if (timerRef.current !== null) window.clearInterval(timerRef.current)
  }

  const handleSubmit = async () => {
    setExpanded(null)
    setPhase('processing')
    setOutcome('pending')
    setErrorMessage(null)
    setElapsedMs(0)
    setShowSlowNotice(false)
    startTimer()

    // A multi-modality request can legitimately take 50+ seconds on a cold free-tier backend; the notice at 15s is
    // informational only and the abort at 90s is a ceiling for a genuinely hung request. The elapsed timer is real.
    const controller = new AbortController()
    abortControllerRef.current = controller
    const slowNoticeTimeoutId = window.setTimeout(() => setShowSlowNotice(true), 15_000)
    const abortTimeoutId = window.setTimeout(() => controller.abort(), 90_000)
    const clearTimers = () => {
      window.clearTimeout(slowNoticeTimeoutId)
      window.clearTimeout(abortTimeoutId)
      stopTimer()
    }

    try {
      // One face capture, both voice sentences, one gesture sample.
      const voice = voiceSentences.filter((s): s is VoiceRecording => s !== null)
      const samples = buildFusionSamples(selected, { ...captured, voice: voice.length ? voice : undefined })
      // The backend authenticates and fuses exactly these modalities.
      const result = await authenticateFusion(userId, APPLICATION_ID, samples, {
        buildingId: building.id,
        signal: controller.signal,
      })
      clearTimers()

      if (result.status === 'ENROLLMENT_REQUIRED') {
        // A submitted factor is not enrolled: nothing was verified - straight to the Enrollment Required screen.
        navigate(`/building/${building.id}/result`, { state: { result } })
        return
      }

      // The meter resolves to the real decision, then the result page tells the full story.
      setOutcome(result.authenticated ? 'granted' : 'denied')
      const finalLatency = Math.round(performance.now() - startedAtRef.current)
      recordAttempt({
        buildingId: building.id,
        modalitiesUsed: result.modalities_used,
        fusionSimilarity: result.fusion_similarity,
        fusionThreshold: result.fusion_threshold,
        authenticated: result.authenticated,
        latencyMs: finalLatency,
      })
      window.setTimeout(() => navigate(`/building/${building.id}/result`, { state: { result } }), reducedMotion ? 300 : 1100)
    } catch (error) {
      clearTimers()
      setPhase('error')
      const timedOut = error instanceof DOMException && error.name === 'AbortError'
      setErrorMessage(
        timedOut
          ? 'Verification timed out. Please try again.'
          : error instanceof ApiError
            ? error.detail
            : 'The backend is unreachable. Is uvicorn running?',
      )
    } finally {
      abortControllerRef.current = null
    }
  }

  const capturedMap = Object.fromEntries(selected.map((m) => [m, isCaptured(m)])) as Partial<Record<Modality, boolean>>
  const capturedCount = selected.filter(isCaptured).length
  const decided = phase === 'processing' && outcome !== 'pending'
  const flow: Step[] = [
    { key: 'select', label: 'Select', state: phase === 'select' ? 'current' : 'done', detail: selected.length ? `${selected.length} factors` : undefined },
    {
      key: 'capture',
      label: 'Capture',
      state: phase === 'select' ? 'upcoming' : phase === 'capture' ? 'current' : phase === 'error' ? 'attention' : 'done',
      detail: phase === 'select' ? undefined : `${capturedCount} / ${selected.length} captured`,
    },
    { key: 'fusion', label: 'Fusion', state: phase === 'processing' ? (decided ? 'done' : 'current') : 'upcoming' },
    { key: 'decision', label: 'Decision', state: decided ? 'current' : 'upcoming' },
  ]

  // What was captured, shown on the factor card (small) and in the expanded panel before a retake (large).
  const summaryFor = (m: Modality, large = false): ReactNode => {
    if (m === 'face' && faceUrl) {
      return <img src={faceUrl} alt="Captured face" className={large ? 'max-h-72 rounded-xl object-contain' : 'h-14 w-20 rounded-lg object-cover'} />
    }
    if (m === 'voice' && voicePrint) {
      return (
        <div className={large ? 'w-full' : 'w-full px-3'}>
          <Voiceprint levels={voicePrint} compact={!large} label="Voiceprint of the recorded sentence" />
        </div>
      )
    }
    if (m === 'hand' && handPath) {
      return (
        <div className={large ? 'h-48 w-48' : 'h-14 w-14'}>
          <GestureTrajectory path={handPath} draw={large} strokeWidth={large ? 3 : 6} label="Captured hand trajectory" />
        </div>
      )
    }
    return null
  }

  const captureFor = (m: Modality): ReactNode => {
    // Camera captures get a comfortable column; voice (two sentence cards) uses the full width.
    if (m === 'face') return <div className="mx-auto max-w-2xl"><FaceCapture mode="verify" onCapture={handleCapture('face')} /></div>
    if (m === 'voice') return <VoiceSentencesCapture mode="verify" onChange={setVoiceSentences} />
    return (
      <div className="mx-auto max-w-2xl">
      <HandGestureCapture
        mode="verify"
        onCapture={(payload) => {
          handleCapture('hand')(payloadBlob(payload), 'gesture.json')
          setHandPath(gestureTrajectoryPath(payload))
        }}
      />
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-5xl px-4 pt-6 pb-14 sm:px-6">
      <header className="mb-6 flex flex-col items-center gap-2 text-center">
        <div className="flex flex-wrap items-center justify-center gap-2">
          <span className="text-eyebrow text-muted-foreground">Identity verification</span>
          <ClearanceBadge level={building.clearanceLevel} />
        </div>
        <h1 className="text-page-title text-foreground">{building.name}</h1>
      </header>

      <ProgressStepper steps={flow} label="Authentication progress" className="mx-auto mb-8 max-w-3xl" />

      <AnimatePresence mode="wait">
        {phase === 'select' && (
          <motion.div key="select" variants={rise} initial="hidden" animate="show" exit="exit" className="mx-auto max-w-3xl">
            <h2 className="text-section-title mb-1 text-center text-foreground">Choose the factors to present</h2>
            <p className="mb-5 text-center text-sm text-muted-foreground">Any of your enrolled factors. They are verified together and fused into one decision.</p>
            <div className="mb-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
              {FACTORS.map((modality) => {
                const isEnrolled = !!enrolled?.[modality]
                const isSelected = selected.includes(modality)
                return (
                  <div key={modality} className="flex flex-col gap-2">
                    <motion.button
                      type="button"
                      disabled={!isEnrolled}
                      aria-pressed={isSelected}
                      onClick={() => toggle(modality)}
                      whileTap={isEnrolled ? { scale: 0.98 } : undefined}
                      className={`relative flex items-center gap-3 rounded-2xl border p-4 text-left transition-colors duration-200 sm:flex-col sm:items-start sm:gap-4 ${
                        isSelected
                          ? 'border-primary/70 bg-primary/10'
                          : isEnrolled
                            ? 'cursor-pointer border-border bg-surface/80 hover:border-border-strong hover:bg-raised/60'
                            : 'cursor-not-allowed border-dashed border-border bg-surface/40 opacity-60'
                      }`}
                    >
                      <ModalityBadge modality={modality} tone={isSelected ? 'active' : isEnrolled ? 'default' : 'muted'} />
                      <span className="min-w-0 flex-1">
                        <span className="text-card-title block text-foreground">{MODALITY_LABEL[modality]}</span>
                        <span className="text-meta block text-muted-foreground">{FACTOR_DETAIL[modality]}</span>
                      </span>
                      <span
                        aria-hidden
                        className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full border transition-colors sm:absolute sm:top-4 sm:right-4 ${
                          isSelected ? 'border-primary bg-primary text-primary-foreground' : 'border-border-strong'
                        }`}
                      >
                        <AnimatePresence>
                          {isSelected && (
                            <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }} transition={{ duration: 0.15 }}>
                              <Check className="h-3 w-3" strokeWidth={3} />
                            </motion.span>
                          )}
                        </AnimatePresence>
                      </span>
                    </motion.button>
                    <div className="flex min-h-6 items-center justify-center gap-2">
                      {profileLoading ? <span className="h-5 w-20 animate-pulse rounded-full bg-raised" aria-hidden /> : <EnrollmentStatusPill status={statuses[modality]} />}
                      {!isEnrolled && !profileLoading && (
                        <Link to={`/building/${building.id}/register?focus=${modality}`} className="text-xs font-medium text-primary hover:underline">
                          Enroll
                        </Link>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
            <button type="button" disabled={selected.length === 0} onClick={() => setPhase('capture')} className="btn btn-primary w-full">
              Begin capture ({selected.length} factor{selected.length === 1 ? '' : 's'})
              <ArrowRight className="h-4 w-4" strokeWidth={1.75} aria-hidden />
            </button>
          </motion.div>
        )}

        {phase === 'capture' && (
          <motion.div key="capture" variants={rise} initial="hidden" animate="show" exit="exit">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <button type="button" onClick={() => setPhase('select')} className="btn btn-ghost -ml-3 min-h-10 px-3 text-xs">
                <ArrowLeft className="h-3.5 w-3.5" strokeWidth={1.5} aria-hidden />
                Change factors
              </button>
              <p className="text-xs text-muted-foreground">Captures stay in this browser until you submit.</p>
            </div>
            {retry && retry.modalities.length > 0 && (
              <p className="mb-4 flex items-start gap-2 rounded-xl border border-warning/35 bg-warning/10 px-4 py-3 text-sm text-foreground">
                <RotateCcw className="mt-0.5 h-4 w-4 shrink-0 text-warning" strokeWidth={1.5} aria-hidden />
                <span>
                  {retry.modalities.map((m) => MODALITY_LABEL[m]).join(' and ')}{' '}
                  {retry.modalities.length === 1 ? 'needs' : 'need'} another attempt. Capture each selected factor again - they
                  are verified together.
                </span>
              </p>
            )}

            <LayoutGroup>
              {/* The expanded factor: its card has grown into the full capture experience (shared layoutId). */}
              <AnimatePresence mode="popLayout">
                {expanded && (
                  <motion.section
                    key={expanded}
                    layoutId={`capture-${expanded}`}
                    style={{ borderRadius: 24 }}
                    transition={{ type: 'spring', stiffness: 260, damping: 32 }}
                    aria-label={`${MODALITY_LABEL[expanded]} capture`}
                    className="surface-card mb-5 border-primary/45 p-3 sm:p-5"
                  >
                    <motion.div layout="position" className="mb-3 flex items-center justify-between gap-3">
                      <span className="flex items-center gap-3">
                        <ModalityBadge modality={expanded} tone="active" />
                        <span>
                          <span className="text-eyebrow block text-muted-foreground">Capturing</span>
                          <span className="text-section-title block text-foreground">{MODALITY_LABEL[expanded]}</span>
                        </span>
                      </span>
                      <button type="button" onClick={closeFactor} className="btn btn-ghost min-h-10 px-3 text-xs" aria-label="Close capture and return to the factor overview">
                        <X className="h-4 w-4" strokeWidth={1.75} aria-hidden />
                        <span className="hidden sm:inline">Back to overview</span>
                      </button>
                    </motion.div>

                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.15, duration: 0.25 } }}>
                      {reviewing && isCaptured(expanded) && !retaking[expanded] ? (
                        <div className="flex flex-col items-center gap-4 rounded-2xl border border-success/30 bg-success/[0.05] p-5">
                          <p className="flex items-center gap-2 text-sm font-medium text-success">
                            <Check className="h-4 w-4" strokeWidth={2.25} aria-hidden /> {MODALITY_LABEL[expanded]} captured
                          </p>
                          {summaryFor(expanded, true)}
                          <button type="button" onClick={() => retake(expanded)} className="btn btn-secondary">
                            <RotateCcw className="h-4 w-4" strokeWidth={1.5} aria-hidden />
                            Retake {MODALITY_LABEL[expanded].toLowerCase()}
                          </button>
                        </div>
                      ) : (
                        captureFor(expanded)
                      )}
                      <div className="mt-4 flex justify-end">
                        <button type="button" onClick={() => continueFrom(expanded)} disabled={!isCaptured(expanded)} className="btn btn-primary w-full sm:w-auto">
                          {selected.some((o) => o !== expanded && !isCaptured(o)) ? 'Continue to next factor' : 'Done'}
                          <ArrowRight className="h-4 w-4" strokeWidth={1.75} aria-hidden />
                        </button>
                      </div>
                    </motion.div>
                  </motion.section>
                )}
              </AnimatePresence>

              {/* The deck: every other presented factor as a card. Open one to capture it. */}
              <div className={`mb-5 grid grid-cols-1 gap-3 ${selected.length > 1 ? 'sm:grid-cols-2 lg:grid-cols-3' : 'mx-auto max-w-sm'}`}>
                {selected
                  .filter((m) => m !== expanded)
                  .map((m) => (
                    <FactorCard key={m} modality={m} captured={isCaptured(m)} summary={summaryFor(m)} onOpen={() => openFactor(m)} />
                  ))}
              </div>
            </LayoutGroup>

            {!expanded && capturedCount === 0 && (
              <p className="mb-5 text-center text-sm text-muted-foreground">Open a factor to start capturing. Each one expands into its own scanner.</p>
            )}

            <VerificationPipeline modalities={selected} captured={capturedMap} phase="capture" className="mb-5" />
            <div className="sticky bottom-20 z-20 lg:bottom-4">
              <motion.button
                type="button"
                onClick={handleSubmit}
                disabled={!allCaptured}
                whileTap={allCaptured ? { scale: 0.99 } : undefined}
                className="btn btn-primary w-full shadow-lg shadow-black/40"
              >
                <ShieldCheck className="h-4 w-4" strokeWidth={1.75} aria-hidden />
                {allCaptured ? 'Verify identity' : `Capture all factors to continue (${capturedCount} / ${selected.length})`}
              </motion.button>
            </div>
          </motion.div>
        )}

        {phase === 'processing' && (
          <motion.div key="processing" variants={rise} initial="hidden" animate="show" exit="exit" className="mx-auto max-w-xl">
            <FusionMeter modalities={selected} outcome={outcome} />
            <div className="mt-6 text-center" aria-live="polite">
              <p className="text-section-title text-foreground">
                {outcome === 'pending' ? 'Running multimodal verification...' : outcome === 'granted' ? 'Decision reached - identity verified' : 'Decision reached - not verified'}
              </p>
              <p className="mt-1 font-mono text-sm tabular-nums text-muted-foreground">{(elapsedMs / 1000).toFixed(2)} s elapsed</p>
              <p className="mx-auto mt-3 max-w-sm text-xs text-muted-foreground">
                One request: embeddings, keyed template protection under the active template set, matching, and fusion - all on the
                backend.
              </p>
              <AnimatePresence>
                {showSlowNotice && (
                  <motion.p variants={rise} initial="hidden" animate="show" exit="exit" className="mt-3 text-xs text-muted-foreground">
                    Still verifying - this can take longer when the backend has been inactive.
                  </motion.p>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}

        {phase === 'error' && (
          <motion.div key="error" variants={rise} initial="hidden" animate="show" exit="exit" className="mx-auto max-w-md text-center">
            <div role="alert" className="surface-card mb-5 flex flex-col items-center gap-2 border-danger/35 p-6">
              <span className="flex h-11 w-11 items-center justify-center rounded-full border border-danger/35 bg-danger/10">
                <AlertCircle className="h-5 w-5 text-danger" strokeWidth={1.5} aria-hidden />
              </span>
              <p className="text-section-title text-foreground">Verification could not be completed</p>
              <p className="text-sm text-muted-foreground">{errorMessage}</p>
            </div>
            <button type="button" onClick={() => setPhase('capture')} className="btn btn-secondary">
              <ArrowLeft className="h-4 w-4" strokeWidth={1.5} aria-hidden />
              Back to capture
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
