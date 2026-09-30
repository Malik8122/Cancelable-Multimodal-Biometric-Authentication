import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import { RotateCcw, UserPlus } from 'lucide-react'
import { Link, useLocation, useParams } from 'react-router-dom'
import type { AuthenticationOutcome, Modality } from '../api/types'
import { AnimatedNumber } from '../components/biometric/AnimatedNumber'
import { FusionReveal } from '../components/biometric/FusionReveal'
import { ResearchDetails } from '../components/biometric/ResearchDetails'
import { useAmbient } from '../hooks/useAmbient'
import { BiometricMetricsPanel } from '../components/biometric/BiometricMetricsPanel'
import { FusionDiagnosticsPanel } from '../components/biometric/FusionDiagnosticsPanel'
import { AttemptAssessmentPanel } from '../components/biometric/AttemptAssessmentPanel'
import { TemplateRotationPanel } from '../components/biometric/TemplateRotationPanel'
import { ModalityOutcomeList } from '../components/biometric/ModalityOutcomeList'
import { MODALITY_LABEL } from '../config/buildings'
import { useBuildings } from '../context/BuildingsContext'
import { useReducedMotion } from '../hooks/useReducedMotion'
import { rise } from '../lib/motion'

const SECONDARY_CLASS = 'btn btn-secondary flex-1'
const PRIMARY_CLASS = 'btn btn-primary flex-1'

// How the presented factors were combined (backend fusion policy), in words.
const POLICY_TEXT: Record<string, string> = {
  ALL_REQUIRED: 'Every presented factor must match',
  AT_LEAST_TWO: 'At least two presented factors must match',
  WEIGHTED: 'Weighted fusion of the presented factors',
}

// One of exactly three states: ACCESS_GRANTED, ACCESS_DENIED, ENROLLMENT_REQUIRED. Enrollment-required is its own
// screen (no verdict animation, no similarity): a factor the user selected is not registered, so nothing was verified.
export function ResultPage() {
  const { buildingId } = useParams<{ buildingId: string }>()
  const location = useLocation() as { state?: { result?: AuthenticationOutcome } }
  const { getBuilding } = useBuildings()
  const building = buildingId ? getBuilding(buildingId) : undefined
  const result = location.state?.result
  const reducedMotion = useReducedMotion()
  // The details appear once the fusion reveal has stated the decision.
  const [revealed, setRevealed] = useState(false)
  const decisionState = result && result.status !== 'ENROLLMENT_REQUIRED' ? result.authentication_state : null
  useAmbient(decisionState === null ? 'idle' : !revealed ? 'processing' : decisionState === 'ACCESS_GRANTED' ? 'success' : 'failure')

  if (!building || !result) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-16 text-center">
        <p className="text-muted-foreground">No authentication result to show.</p>
        <Link to="/" className="mt-4 inline-block text-sm text-primary hover:underline">
          Return to the campus
        </Link>
      </div>
    )
  }

  if (result.status === 'ENROLLMENT_REQUIRED') {
    const missing = result.missing_modalities
    const names = missing.map((m) => MODALITY_LABEL[m])
    const lower = names.map((n) => n.toLowerCase())
    return (
      <div className="mx-auto max-w-2xl px-6 pt-10 pb-32">
        <p className="mb-2 text-center text-xs font-medium tracking-wide text-muted-foreground">{building.name}</p>
        <div className="flex flex-col items-center py-12 text-center">
          <div className="mb-6 flex h-24 w-24 items-center justify-center rounded-full border border-warning/30 bg-warning/10">
            <UserPlus className="h-10 w-10 text-warning" strokeWidth={1.5} />
          </div>
          <h1 className="mb-3 text-3xl font-semibold tracking-tight text-warning sm:text-4xl">Enrollment Required</h1>
          <p className="max-w-md text-sm text-muted-foreground">
            You selected {names.join(' and ')}, which {missing.length === 1 ? 'is' : 'are'} not registered. Please complete{' '}
            {lower.join(' and ')} enrollment before requesting access with {missing.length === 1 ? 'it' : 'them'}.
          </p>
        </div>
        <div className="flex flex-col gap-3 sm:flex-row">
          <Link to={`/building/${building.id}/register?focus=${missing[0]}`} className={PRIMARY_CLASS}>
            Go to Enrollment
          </Link>
          <Link to={`/building/${building.id}/authenticate`} className={SECONDARY_CLASS}>
            Choose Other Factors
          </Link>
        </div>
      </div>
    )
  }

  const granted = result.authentication_state === 'ACCESS_GRANTED'
  const retry = result.attempt_assessment?.status === 'RETRY_REQUESTED' ? result.attempt_assessment : undefined
  const reveal = { hidden: {}, show: { transition: { staggerChildren: 0.07, delayChildren: reducedMotion ? 0 : 0.5 } } }

  return (
    <div className="mx-auto max-w-2xl px-4 pt-6 pb-14 sm:px-6">
      <p className="text-eyebrow text-center text-muted-foreground">{building.name}</p>

      <FusionReveal result={result} onRevealed={() => setRevealed(true)} />

      <AnimatePresence>
      {revealed && (
      <motion.div variants={reveal} initial="hidden" animate="show" className="space-y-4">
        <motion.section variants={rise} className="surface-card overflow-hidden" aria-labelledby="factors-title">
          <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-border px-4 py-3 sm:px-5">
            <h2 id="factors-title" className="text-card-title text-foreground">
              Presented factors
            </h2>
            <p className="text-meta text-muted-foreground">{POLICY_TEXT[result.fusion_policy] ?? result.fusion_policy}</p>
          </div>
          <ModalityOutcomeList result={result} />
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 border-t border-border px-4 py-3 text-xs sm:grid-cols-3 sm:px-5">
            <div>
              <dt className="text-muted-foreground">Template set</dt>
              <dd className="font-medium text-foreground">Set {result.active_template_set}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Verification time</dt>
              <dd className="font-mono font-medium tabular-nums text-foreground">
                <AnimatedNumber value={Math.round(result.authentication_time_ms)} suffix=" ms" />
              </dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Decision</dt>
              <dd className={`font-medium ${granted ? 'text-success' : 'text-danger'}`}>{granted ? 'Authentication complete' : 'Access denied'}</dd>
            </div>
          </dl>
        </motion.section>

        <motion.div variants={rise}>
          <AttemptAssessmentPanel result={result} />
          <TemplateRotationPanel result={result} />
        </motion.div>

        <motion.div variants={rise}>
          <BiometricMetricsPanel result={result} />
          <div className="mb-4">
            <ResearchDetails result={result} />
          </div>
          <FusionDiagnosticsPanel result={result} />
        </motion.div>

        <motion.div variants={rise} className="flex flex-col gap-3 pt-2 sm:flex-row">
          <Link
            to={`/building/${building.id}/authenticate`}
            // A retry goes straight back to capture with the same factors, naming the ones the backend asked for again.
            state={retry ? { retry: { factors: Object.keys(retry.modality_statuses) as Modality[], modalities: retry.retry_modalities } } : undefined}
            className={`${granted ? SECONDARY_CLASS : PRIMARY_CLASS}`}
          >
            <RotateCcw className="h-4 w-4" strokeWidth={1.5} aria-hidden />
            {granted ? 'Authenticate again' : 'Retry'}
          </Link>
          {granted && (
            <Link to="/templates" className={SECONDARY_CLASS}>
              Manage template sets
            </Link>
          )}
          <Link to="/" className={SECONDARY_CLASS}>
            Return to campus
          </Link>
        </motion.div>
      </motion.div>
      )}
      </AnimatePresence>
    </div>
  )
}
