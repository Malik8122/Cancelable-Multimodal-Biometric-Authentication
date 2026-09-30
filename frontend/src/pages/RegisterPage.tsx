import { AnimatePresence, motion } from 'motion/react'
import { AlertCircle, ArrowRight, Check, Loader2, UserRound } from 'lucide-react'
import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { createUser, enroll, enrollFace, enrollHand, setDisplayName } from '../api/client'
import { ApiError, type EnrollResponse, type EnrollmentStatus, type FacePose, type RecordingQuality } from '../api/types'
import { EnrollmentStatusPill } from '../components/biometric/EnrollmentStatusPill'
import { ModalityBadge } from '../components/biometric/ModalityBadge'
import { ProgressStepper, type Step } from '../components/biometric/ProgressStepper'
import { FAIR_MESSAGE, POOR_MESSAGE, RecordingQualityPanel } from '../components/biometric/RecordingQualityPanel'
import { TemplateSetAnimation, animationTicks } from '../components/biometric/TemplateSetAnimation'
import { GuidedFaceCapture, FACE_POSE_ORDER } from '../components/capture/GuidedFaceCapture'
import { HandGestureEnrollment } from '../components/capture/HandGestureEnrollment'
import { VoiceSentencesCapture, type VoiceRecording, type VoiceSentences } from '../components/capture/VoiceSentencesCapture'
import { emptyVoiceSlots } from '../config/protocol'
import { APPLICATION_ID } from '../config/app'
import { MODALITY_LABEL } from '../config/buildings'
import { useBuildings } from '../context/BuildingsContext'
import { useSession } from '../context/SessionContext'
import { useEnrollmentProfile } from '../hooks/useEnrollmentProfile'
import { useAmbient } from '../hooks/useAmbient'
import {
  FACE_ENROLLMENT_SAMPLES,
  HAND_ENROLLMENT_SAMPLES,
  REGISTRATION_STEPS,
  registrationStepLabel,
  VOICE_ENROLLMENT_PROMPTS,
} from '../config/protocol'
import { rise } from '../lib/motion'

type CardModality = 'face' | 'voice' | 'hand'

interface SubmitResult {
  message: string
  warning?: boolean
  /** Voice: the recording quality band to show ("Recording Quality: Excellent / Good / Fair"). */
  quality?: RecordingQuality
}

// Voice enrollment feedback that is not a plain error: a FAIR warning (continue or re-record) or a POOR rejection.
interface QualityNotice {
  quality: 'FAIR' | 'POOR'
  /** Development only: the backend's consistency similarity of the two recordings (DEBUG_SCORES). */
  similarity?: number
}

// Mirrors backend/display_names.py (the backend re-validates; this only gives instant feedback).
const MAX_NAME_LENGTH = 64
const normalizeName = (raw: string) => raw.trim().split(/\s+/).join(' ')
function validateName(raw: string): string | null {
  const name = normalizeName(raw)
  if (!name) return 'Please enter your name.'
  if (name.length > MAX_NAME_LENGTH) return `The name must be at most ${MAX_NAME_LENGTH} characters.`
  if (!/^\p{L}/u.test(name)) return 'The name must start with a letter.'
  if (!/^[\p{L} '\u2019.-]+$/u.test(name)) return 'The name may only contain letters, spaces, apostrophes, hyphens and periods.'
  return null
}

// Account setup (before the numbered biometric steps). A new registration creates the backend user here (POST /users generates the internal id - the name is only
// a label, so two people may share one). A user registered before names existed can name their account the same way.
function NameStep({
  mode,
  onSaved,
  initialName = '',
  onCancel,
}: {
  mode: 'new' | 'legacy' | 'edit'
  onSaved: (name: string) => Promise<void>
  initialName?: string
  onCancel?: () => void
}) {
  const [name, setName] = useState(initialName)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    const problem = validateName(name)
    if (problem) {
      setError(problem)
      return
    }
    setBusy(true)
    setError(null)
    try {
      await onSaved(normalizeName(name))
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : 'The backend is unreachable.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit} className="surface-card border-primary/45 p-4 sm:p-5">
      <div className="mb-4 flex items-center gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-primary/25 bg-primary/10">
          <UserRound className="h-5 w-5 text-primary" strokeWidth={1.5} aria-hidden />
        </div>
        <div>
          <p className="text-eyebrow text-muted-foreground">{mode === 'edit' ? 'Profile' : 'Before you begin'}</p>
          <h2 className="text-card-title text-foreground">{mode === 'edit' ? 'Edit user' : 'Your name'}</h2>
          <p className="text-xs text-muted-foreground">
            {mode === 'new'
              ? 'Enter the name to show for this person. It is only a label - it is never part of the biometric template.'
              : mode === 'edit'
                ? 'Change the display name. It is only a label - the biometric templates stay exactly as they are.'
                : 'This account was registered before names were added. Give it a name so it is easy to recognise.'}
          </p>
        </div>
      </div>
      <label htmlFor="display-name" className="mb-1.5 block text-xs font-medium text-muted-foreground">
        Full Name
      </label>
      <input
        id="display-name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        autoComplete="name"
        placeholder="e.g. Sanya Malik"
        className="w-full rounded-lg border border-border-strong bg-raised px-3 py-2.5 text-sm text-foreground placeholder:text-muted-foreground"
      />
      {error && (
        <p className="mt-2 flex items-start gap-2 text-xs text-danger">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.5} /> {error}
        </p>
      )}
      <button
        type="submit"
        disabled={busy}
        className="btn btn-primary mt-4 w-full"
      >
        {busy ? 'Saving…' : mode === 'new' ? 'Continue to Face Capture' : 'Save Name'}
      </button>
      {onCancel && (
        <button type="button" onClick={onCancel} className="btn btn-secondary mt-2 w-full text-muted-foreground">
          Cancel
        </button>
      )}
    </form>
  )
}

const describeEnrollment = (r: EnrollResponse) =>
  `${r.templates_created} template sets generated - Set ${r.active_template_set_version} is active.`

// One independent enrollment card. Enrolling (or re-enrolling) one modality never touches the others: the backend
// adds this modality's template to every live template set and leaves the other modalities' templates alone.
function EnrollmentCard({
  modality,
  step,
  title,
  blurb,
  status,
  highlighted,
  ready,
  poolSize,
  onSubmit,
  onDone,
  onFailed,
  resetKey,
  children,
}: {
  modality: CardModality
  step: string
  title: string
  blurb: string
  status: EnrollmentStatus | undefined
  highlighted: boolean
  ready: boolean
  poolSize: number
  onSubmit: (acceptLowQuality?: boolean) => Promise<SubmitResult>
  onDone: () => void
  onFailed: () => void
  resetKey: number
  children: ReactNode
}) {
  const [open, setOpen] = useState(highlighted)
  const [busy, setBusy] = useState(false)
  const [tick, setTick] = useState(0)
  const [result, setResult] = useState<SubmitResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<QualityNotice | null>(null)
  const cardRef = useRef<HTMLDivElement>(null)
  const enrolled = status === 'REGISTERED' || status === 'UPDATED'

  useEffect(() => {
    if (highlighted) cardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [highlighted])

  useEffect(() => {
    if (!busy) return
    setTick(0)
    // Illustrative: capture -> template set 1..N -> pool. It holds on the last step until the real response arrives.
    const interval = window.setInterval(() => setTick((t) => Math.min(t + 1, animationTicks(poolSize) - 1)), 350)
    return () => window.clearInterval(interval)
  }, [busy, poolSize])

  const submit = async (acceptLowQuality = false) => {
    setBusy(true)
    setError(null)
    setResult(null)
    setNotice(null)
    try {
      const outcome = await onSubmit(acceptLowQuality)
      setTick(animationTicks(poolSize))
      setResult(outcome)
      setOpen(false)
      onDone()
    } catch (err) {
      const body = err instanceof ApiError ? (err.body as { status?: string; similarity?: number } | undefined) : undefined
      if (body?.status === 'LOW_QUALITY_WARNING') {
        // Fair: nothing is stored yet. Keep the recordings so "Continue Enrollment" can use them, or "Re-record".
        setNotice({ quality: 'FAIR', similarity: body.similarity })
      } else if (body?.status === 'ENROLLMENT_INCONSISTENT') {
        // Poor: rejected, nothing stored. Clear the recordings so the user records again.
        setNotice({ quality: 'POOR', similarity: body.similarity })
        onFailed()
      } else {
        // Nothing was stored (e.g. no face detected): drop the captured samples and refresh the status.
        setError(err instanceof ApiError ? err.detail : 'The backend is unreachable.')
        onFailed()
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <motion.div
      ref={cardRef}
      layout
      aria-labelledby={`enroll-${modality}-title`}
      className={`surface-card p-4 transition-colors duration-300 sm:p-5 ${open || highlighted ? 'border-primary/45' : enrolled ? 'border-success/30' : ''}`}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-3">
          <ModalityBadge modality={modality} tone={enrolled ? 'success' : open ? 'active' : 'default'} />
          <div className="min-w-0">
            <p className="text-eyebrow text-muted-foreground">{step}</p>
            <h2 id={`enroll-${modality}-title`} className="text-card-title text-foreground">
              {title}
            </h2>
            <p className="mt-0.5 max-w-lg text-meta text-muted-foreground">{blurb}</p>
          </div>
        </div>
        <EnrollmentStatusPill status={status} />
      </div>

      {result?.quality && (
        <div className="mt-4">
          <RecordingQualityPanel quality={result.quality} message={result.quality === 'FAIR' ? FAIR_MESSAGE : undefined} />
          <p className="-mt-2 text-xs text-success">{result.message}</p>
        </div>
      )}

      {result && !result.quality && (
        <motion.p
          variants={rise}
          initial="hidden"
          animate="show"
          role="status"
          className={`mt-4 flex items-start gap-2 rounded-lg border p-3 text-xs ${
            result.warning ? 'border-warning/30 bg-warning/10 text-warning' : 'border-success/30 bg-success/10 text-success'
          }`}
        >
          {result.warning ? <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.5} /> : <Check className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={2} />}
          {result.message}
        </motion.p>
      )}

      {!open && !busy && (
        <div className="mt-4 flex flex-wrap gap-2">
          {enrolled ? (
            // One action for an enrolled modality: opens the same enrollment flow (replaces this modality only).
            <button
              type="button"
              onClick={() => {
                setResult(null)
                setOpen(true)
              }}
              className="btn btn-secondary text-xs"
            >
              Re-enroll
            </button>
          ) : (
            <button type="button" onClick={() => setOpen(true)} className="btn btn-primary text-xs">
              {status === 'RETRY_REQUIRED' ? 'Retry Enrollment' : `Enroll ${MODALITY_LABEL[modality]}`}
            </button>
          )}
        </div>
      )}

      <AnimatePresence initial={false}>
      {open && !busy && (
        <motion.div
          key={resetKey}
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.25 }}
          className="mt-4 overflow-hidden"
        >
          {enrolled && (
            <p className="mb-3 text-xs text-muted-foreground">
              Your new sample replaces the existing {MODALITY_LABEL[modality].toLowerCase()} templates. Your other biometrics are not affected.
            </p>
          )}
          {notice?.quality === 'FAIR' && (
            <RecordingQualityPanel quality="FAIR" message={FAIR_MESSAGE}>
              <button type="button" onClick={() => void submit(true)} className="btn btn-primary text-xs">
                Continue Enrollment
              </button>
              <button
                onClick={() => {
                  setNotice(null)
                  onFailed() // clear both recordings and remount the recorders
                }}
                type="button"
                className="btn btn-secondary text-xs"
              >
                Re-record
              </button>
            </RecordingQualityPanel>
          )}
          {notice?.quality === 'POOR' && (
            <RecordingQualityPanel
              quality="POOR"
              message={`${POOR_MESSAGE}${notice.similarity !== undefined ? ` [dev: consistency ${notice.similarity.toFixed(3)} - enrolls at >= 0.60, recommended >= 0.75]` : ''}`}
            >
              <button type="button" onClick={() => setNotice(null)} className="btn btn-primary text-xs">
                Record Again
              </button>
            </RecordingQualityPanel>
          )}
          {children}
          {error && (
            <p role="alert" className="mt-3 flex items-start gap-2 rounded-lg border border-danger/30 bg-danger/10 p-2.5 text-xs text-danger">
              <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.5} aria-hidden /> {error}
            </p>
          )}
          <div className="mt-4 flex gap-2">
            <button type="button" onClick={() => void submit(false)} disabled={!ready} className="btn btn-primary flex-1">
              {enrolled ? `Update ${MODALITY_LABEL[modality]} Enrollment` : `Enroll ${MODALITY_LABEL[modality]}`}
            </button>
            <button type="button" onClick={() => setOpen(false)} className="btn btn-secondary text-muted-foreground">
              Cancel
            </button>
          </div>
        </motion.div>
      )}
      </AnimatePresence>

      {busy && (
        <div className="mt-5">
          <TemplateSetAnimation modalities={[modality]} poolSize={poolSize} tick={tick} />
          <p role="status" className="mt-3 flex items-center justify-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" strokeWidth={1.5} aria-hidden /> Generating protected template sets&hellip;
          </p>
        </div>
      )}
    </motion.div>
  )
}

export function RegisterPage() {
  const { buildingId } = useParams<{ buildingId: string }>()
  const [searchParams, setSearchParams] = useSearchParams()
  const focus = searchParams.get('focus')
  const startNew = searchParams.get('new') === '1'
  const { userId, health, addRegisteredUser, rememberName } = useSession()
  const { getBuilding } = useBuildings()
  const building = buildingId ? getBuilding(buildingId) : undefined
  const poolSize = health?.template_pool_size ?? 4
  const { statuses, enrolledList, displayName, hasDisplayName, profileEditingEnabled, loading, error: loadError, refresh } =
    useEnrollmentProfile(userId)
  const [editingName, setEditingName] = useState(false)

  // A new registration must start with a name. The current id counts as "not registered yet" when it has neither a
  // name nor any enrollment (the placeholder id a first visit gets); it is then replaced by the new backend user.
  const unregistered = !loading && !loadError && !hasDisplayName && enrolledList.length === 0
  const needsName = startNew || unregistered
  const legacyUnnamed = !startNew && !loading && !loadError && !hasDisplayName && enrolledList.length > 0
  const isEnrolled = (s: EnrollmentStatus | undefined) => s === 'REGISTERED' || s === 'UPDATED'
  const registrationComplete = !needsName && isEnrolled(statuses.face) && isEnrolled(statuses.voice)
  useAmbient(registrationComplete ? 'success' : 'idle')

  const registerName = async (name: string) => {
    const created = await createUser(name)
    addRegisteredUser(created.user_id, created.display_name, unregistered ? userId : undefined)
    setSearchParams({}, { replace: true })
  }
  const nameExistingUser = async (name: string) => {
    const saved = await setDisplayName(userId, name)
    rememberName(saved.user_id, saved.display_name)
    await refresh()
  }

  const [facePoses, setFacePoses] = useState<Partial<Record<FacePose, Blob>>>({})
  const [handAttempts, setHandAttempts] = useState<Blob[] | null>(null)
  // Voice ENROLLMENT: one slot per enrollment prompt (config/protocol.ts).
  const [voice, setVoice] = useState<VoiceSentences>(() => emptyVoiceSlots<VoiceRecording>('register'))
  const voiceReady = voice.length > 0 && voice.every(Boolean)
  const [resetKey, setResetKey] = useState(0)

  const clearVoice = () => {
    setVoice(emptyVoiceSlots<VoiceRecording>('register'))
    setResetKey((k) => k + 1)
  }
  const done = () => {
    setFacePoses({})
    setHandAttempts(null)
    clearVoice()
    void refresh()
  }

  const stepState = (s: EnrollmentStatus | undefined, optional = false): Step['state'] =>
    isEnrolled(s) ? 'done' : s === 'RETRY_REQUIRED' ? 'attention' : optional ? 'upcoming' : 'current'
  const stepDetail = (s: EnrollmentStatus | undefined, pending: string) =>
    isEnrolled(s) ? 'Complete' : s === 'RETRY_REQUIRED' ? 'Re-record needed' : pending
  // The three numbered BIOMETRIC steps, in protocol order (config/protocol.ts): 1 Face, 2 Voice, 3 Dynamic Hand Gesture.
  // Naming the user is account setup and is not counted as a step.
  const pending: Record<(typeof REGISTRATION_STEPS)[number]['modality'], string> = {
    face: `${FACE_ENROLLMENT_SAMPLES} samples`,
    voice: `${VOICE_ENROLLMENT_PROMPTS.length} sentences`,
    hand: `Optional - ${HAND_ENROLLMENT_SAMPLES} gesture samples`,
  }
  const setupSteps: Step[] = REGISTRATION_STEPS.map(({ modality, optional }, i) => {
    const previousDone = i === 0 || isEnrolled(statuses[REGISTRATION_STEPS[i - 1].modality]) || isEnrolled(statuses[modality])
    return {
      key: modality,
      label: modality === 'hand' ? 'Hand Gesture' : MODALITY_LABEL[modality],
      state: needsName ? 'upcoming' : optional ? stepState(statuses[modality], true) : previousDone ? stepState(statuses[modality]) : 'upcoming',
      detail: stepDetail(statuses[modality], pending[modality]),
    }
  })

  return (
    <div className="mx-auto max-w-3xl px-4 pt-6 pb-14 sm:px-6">
      {building && <p className="text-eyebrow mb-2 text-center text-muted-foreground">{building.name}</p>}
      <h1 className="text-page-title mb-2 text-center text-foreground">
        {needsName ? 'Register New User' : 'Set Up Your Biometric Identity'}
      </h1>
      <p className="mx-auto mb-6 max-w-xl text-center text-sm text-muted-foreground">
        {needsName
          ? 'Enter your name, capture your face, record your voice - then you are registered.'
          : `Registering ${displayName ?? '...'}. Capture your face, then record your voice. You choose which registered factors to use each time you authenticate.`}
      </p>

      <ProgressStepper steps={setupSteps} label="Biometric registration steps" className="surface-card mb-8 px-4 py-4 sm:px-5" />

      {!needsName && hasDisplayName && profileEditingEnabled && (
        <div className="mx-auto mb-6 max-w-xl">
          {editingName ? (
            <>
              <p className="mb-2 rounded-lg border border-primary/30 bg-primary/10 p-3 text-xs text-muted-foreground">
                <span className="font-medium text-foreground">Edit Profile</span> changes only the display name. The face
                and voice templates, their key versions and the ACTIVE / STANDBY / REVOKED sets are not
                modified. To replace a biometric, use <span className="text-foreground">Re-enroll</span> on its card below.
              </p>
              <NameStep
                mode="edit"
                initialName={displayName ?? ''}
                onSaved={async (name) => {
                  await nameExistingUser(name)
                  setEditingName(false)
                }}
                onCancel={() => setEditingName(false)}
              />
            </>
          ) : (
            <button
              onClick={() => setEditingName(true)}
              type="button"
              className="btn btn-ghost mx-auto flex w-fit text-xs"
            >
              <UserRound className="h-3.5 w-3.5" strokeWidth={1.5} aria-hidden /> Edit User
            </button>
          )}
        </div>
      )}

      {registrationComplete && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          role="status"
          className="mb-6 rounded-2xl border border-success/30 bg-success/10 p-5 text-center"
        >
          <p className="flex items-center justify-center gap-2 text-base font-medium text-success">
            <Check className="h-5 w-5" strokeWidth={2} /> Registration Successful
          </p>
          <p className="mt-1 text-sm text-foreground">Welcome, {displayName}.</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Your face and voice biometric templates have been securely enrolled. Only protected, cancelable templates are
            stored - never your images or recordings.
          </p>
        </motion.div>
      )}

      {loadError && (
        <p className="mb-4 flex items-center gap-2 rounded-lg border border-danger/30 bg-danger/10 p-3 text-sm text-danger">
          <AlertCircle className="h-4 w-4 shrink-0" strokeWidth={1.5} /> {loadError}
        </p>
      )}

      {needsName ? (
        loading ? (
          <p className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} /> Loading&hellip;
          </p>
        ) : (
          <NameStep mode="new" onSaved={registerName} />
        )
      ) : (
        <div className="space-y-4">
          {legacyUnnamed && <NameStep mode="legacy" onSaved={nameExistingUser} />}
          <EnrollmentCard
            modality="face"
            step={registrationStepLabel('face')}
            title="Face"
            blurb="One-time guided enrollment: five full-face captures, all facing the camera."
            status={statuses.face}
            highlighted={focus === 'face'}
            ready={FACE_POSE_ORDER.every((pose) => facePoses[pose])}
            poolSize={poolSize}
            resetKey={resetKey}
            onDone={done}
            onFailed={() => {
              setFacePoses({})
              setResetKey((k) => k + 1)
              void refresh()
            }}
            onSubmit={async () => {
              // All five poses go in ONE request. The backend averages the valid embeddings into a centroid, discards them and
              // generates T1-T4 from the centroid alone; only the protected templates are stored.
              const response = await enrollFace(userId, APPLICATION_ID, facePoses as Record<FacePose, Blob>)
              return { message: `${describeEnrollment(response)} Enrolled from ${response.poses_valid ?? 5} guided poses.` }
            }}
          >
            <GuidedFaceCapture onChange={setFacePoses} />
          </EnrollmentCard>

          <EnrollmentCard
            modality="voice"
            step={registrationStepLabel('voice')}
            title="Voice"
            blurb={`Record the ${VOICE_ENROLLMENT_PROMPTS.length} enrollment sentences, 3-5 seconds each. Each is quality-checked, then they are combined into one voice template. (Verification later asks for two sentences as well.)`}
            status={statuses.voice}
            highlighted={focus === 'voice'}
            ready={voiceReady}
            poolSize={poolSize}
            resetKey={resetKey}
            onDone={done}
            onFailed={() => {
              clearVoice()
              void refresh()
            }}
            onSubmit={async (acceptLowQuality) => {
              // ONE atomic request. The backend compares the two recordings (ECAPA embedding cosine similarity) BEFORE
              // storing anything: Excellent / Good -> enrolled; Fair -> 409 LOW_QUALITY_WARNING (stored only if the user
              // continues); Poor -> 422 ENROLLMENT_INCONSISTENT (nothing stored).
              // The backend's voice enrollment takes the first sentence as the sample and the second as its consistency
              // check (image + confirm_image) - config/protocol.ts keeps VOICE_ENROLLMENT_PROMPTS in step with that.
              if (voice.length !== 2) throw new Error('Voice enrollment on this backend takes two sentences (sample + confirmation).')
              const [primary, confirmation] = voice as [NonNullable<VoiceSentences[number]>, NonNullable<VoiceSentences[number]>]
              const response = await enroll('voice', userId, APPLICATION_ID, primary.blob, primary.filename, {
                confirm: { sample: confirmation.blob, filename: confirmation.filename },
                acceptLowQuality,
              })
              return { message: describeEnrollment(response), quality: response.recording_quality }
            }}
          >
            <VoiceSentencesCapture mode="register" onChange={setVoice} />
          </EnrollmentCard>

          <EnrollmentCard
            modality="hand"
            step={registrationStepLabel('hand')}
            title="Dynamic Hand Gesture"
            blurb={`Capture ${HAND_ENROLLMENT_SAMPLES} independent Z gesture samples to learn your natural movement pattern. Only hand landmarks are sent - never video.`}
            status={statuses.hand}
            highlighted={focus === 'hand'}
            ready={!!handAttempts}
            poolSize={poolSize}
            resetKey={resetKey}
            onDone={done}
            onFailed={() => {
              setHandAttempts(null)
              setResetKey((k) => k + 1)
              void refresh()
            }}
            onSubmit={async () => {
              // All three independent samples in ONE request; the backend re-checks each (capture gate + Z check), refuses
              // duplicates and a sample unlike both others, keeps each as its own sequence, and stores nothing if one fails.
              const response = await enrollHand(userId, APPLICATION_ID, handAttempts!)
              return { message: `${describeEnrollment(response)} Enrolled from ${response.hand_gesture?.attempts ?? HAND_ENROLLMENT_SAMPLES} independent Z gesture samples.` }
            }}
          >
            <HandGestureEnrollment onChange={setHandAttempts} resetKey={resetKey} />
          </EnrollmentCard>
        </div>
      )}

      {building && enrolledList.length > 0 && (
        <div className="mt-8 text-center">
          <Link to={`/building/${building.id}`} className="btn btn-primary px-8">
            Continue to {building.name}
            <ArrowRight className="h-4 w-4" strokeWidth={1.75} aria-hidden />
          </Link>
        </div>
      )}
    </div>
  )
}
