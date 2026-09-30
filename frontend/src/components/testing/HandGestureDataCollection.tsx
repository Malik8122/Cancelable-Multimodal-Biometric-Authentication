import { Download } from 'lucide-react'
import { useState } from 'react'
import { HAND_PIPELINE_VERSION } from '../../config/hand'
import { HAND_ENROLLMENT_SAMPLES } from '../../config/protocol'
import type { GestureCapturePayload } from '../../hand/landmarker'
import { HandGestureCapture } from '../capture/HandGestureCapture'

const PARTICIPANT = /^P\d{3,}$/

/** What a recorded sample is for (evaluation/hand_gesture_evaluation.py folder layout). */
type SampleType = 'enrollment' | 'genuine' | 'impostor'

const SAMPLE_TYPES: { value: SampleType; label: string; hint: string }[] = [
  { value: 'enrollment', label: 'Genuine enrollment', hint: `this participant's ${HAND_ENROLLMENT_SAMPLES} enrollment Z samples` },
  { value: 'genuine', label: 'Genuine authentication', hint: 'this participant verifying against their own enrollment (later / another day)' },
  { value: 'impostor', label: 'Impostor attempt', hint: "this participant's own natural Z, compared only with OTHER participants' enrollments" },
]

function folderFor(type: SampleType, session: number): string {
  return type === 'enrollment' ? 'enroll' : type === 'genuine' ? `session_${session}` : `impostor_${session}`
}

/**
 * Real-data collection for the Z hand gesture (evaluation/hand_gesture_evaluation.py). Each recorded sample is saved as
 * a local JSON file named after its place in the harness layout (`<participant>/<folder>/attempt_<n>.json`): the
 * MediaPipe landmark sequence the backend would receive (never video) plus collection metadata (pseudonymous
 * participant ID, sample type, tracking rate, duration, pipeline version). Nothing is uploaded or stored by the
 * backend. The landmark sequence is kept (not only features) so the data can be re-processed when the pipeline
 * version changes - keep the files on this machine, access-restricted, and delete them when the study ends. Collect
 * only with informed consent, recorded in consent.csv.
 */
export function HandGestureDataCollection() {
  const [participant, setParticipant] = useState('P001')
  const [type, setType] = useState<SampleType>('enrollment')
  const [session, setSession] = useState(1)
  const [attempt, setAttempt] = useState(1)
  const [consent, setConsent] = useState(false)
  const [saved, setSaved] = useState<string[]>([])
  const [resetKey, setResetKey] = useState(0)
  const folder = folderFor(type, session)
  const enrollmentFull = type === 'enrollment' && attempt > HAND_ENROLLMENT_SAMPLES
  const valid = PARTICIPANT.test(participant) && !enrollmentFull

  const reset = (next: Partial<{ type: SampleType; session: number }>) => {
    if (next.type) setType(next.type)
    if (next.session) setSession(next.session)
    setAttempt(1)
  }

  const save = (payload: GestureCapturePayload) => {
    const frames = payload.frames
    const tracked = frames.filter((f) => f.landmarks).length
    const seconds = frames.length > 1 ? (frames[frames.length - 1].t - frames[0].t) / 1000 : 0
    const record = {
      ...payload,
      collection: {
        participant_id: participant,
        sample_type: type,
        folder,
        attempt,
        consent_confirmed: true,
        pipeline_version: HAND_PIPELINE_VERSION,
        client_tracking_fps: seconds > 0 ? Math.round((tracked / seconds) * 10) / 10 : 0,
        client_duration_s: Math.round(seconds * 1000) / 1000,
        frames_with_hand: tracked,
        recorded_on: new Date().toISOString().slice(0, 10),
      },
    }
    const name = `${participant}__${folder}__attempt_${attempt}.json`
    const url = URL.createObjectURL(new Blob([JSON.stringify(record)], { type: 'application/json' }))
    const link = document.createElement('a')
    link.href = url
    link.download = name
    link.click()
    URL.revokeObjectURL(url)
    setSaved((prev) => [`${name}  (${record.collection.client_tracking_fps} fps, ${record.collection.client_duration_s} s)`, ...prev].slice(0, 12))
    setAttempt((n) => n + 1)
    window.setTimeout(() => setResetKey((k) => k + 1), 600)
  }

  const input = 'mt-1 w-full rounded-lg border border-border-strong bg-raised px-3 py-2 font-mono text-sm text-foreground'
  return (
    <div className="surface-card p-5">
      <p className="mb-4 text-[13px] leading-relaxed text-muted-foreground">
        Records REAL Z gesture samples for calibrating the hand threshold. Per participant: {HAND_ENROLLMENT_SAMPLES} genuine
        enrollment samples, then several genuine authentication samples (ideally on another day), and impostor samples
        (their own natural Z, scored only against other participants). Files are saved on this computer only (landmarks
        + metadata, no video); file them with <span className="font-mono break-all">python -m evaluation.hand_gesture_evaluation --root DATA --import-downloads DOWNLOADS</span>{' '}
        and evaluate with <span className="font-mono break-all">--root DATA --consent-confirmed</span>.
      </p>
      <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-4">
        <label className="text-xs text-muted-foreground">
          Participant (pseudonymous)
          <input value={participant} onChange={(e) => { setParticipant(e.target.value.trim()); reset({}) }} className={input} />
        </label>
        <label className="text-xs text-muted-foreground sm:col-span-2">
          Sample type
          <select value={type} onChange={(e) => reset({ type: e.target.value as SampleType })} className={input}>
            {SAMPLE_TYPES.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
          </select>
        </label>
        <label className="text-xs text-muted-foreground">
          {type === 'enrollment' ? 'Next sample' : 'Session / attempt'}
          <div className="flex gap-2">
            {type !== 'enrollment' && (
              <input type="number" min={1} value={session} onChange={(e) => reset({ session: Math.max(1, Number(e.target.value) || 1) })} className={input} aria-label="Session" />
            )}
            <input type="number" min={1} value={attempt} onChange={(e) => setAttempt(Math.max(1, Number(e.target.value) || 1))} className={input} aria-label="Attempt" />
          </div>
        </label>
      </div>
      <p className="mb-3 text-xs text-muted-foreground">
        {SAMPLE_TYPES.find((t) => t.value === type)?.hint} - saved as <span className="font-mono">{participant}/{folder}/attempt_{attempt}.json</span>
        {enrollmentFull && <span className="text-warning"> - enrollment already has {HAND_ENROLLMENT_SAMPLES} samples; switch to another sample type.</span>}
      </p>
      <label className="mb-4 flex items-center gap-2 text-xs text-foreground">
        <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
        This participant gave informed consent and is listed in consent.csv.
      </label>
      <HandGestureCapture mode="register" attemptLabel={`${participant} / ${folder} / attempt ${attempt}`} onCapture={save}
        disabled={!consent || !valid} resetKey={resetKey} />
      {saved.length > 0 && (
        <ul className="mt-3 space-y-1 text-xs text-muted-foreground">
          {saved.map((name) => (
            <li key={name} className="flex items-center gap-1.5 font-mono"><Download className="h-3 w-3" strokeWidth={1.5} />{name}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
