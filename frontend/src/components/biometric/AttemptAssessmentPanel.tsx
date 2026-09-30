import { RotateCcw, ShieldAlert } from 'lucide-react'
import type { AuthenticationDecision, Modality } from '../../api/types'
import { MODALITY_LABEL } from '../../config/buildings'

const ORDER: Modality[] = ['face', 'voice', 'hand']

// The backend's handling of a DENIED attempt that did NOT rotate templates: retry guidance (capture error, insufficient
// quality, or an isolated mismatch) - or a suspicious attempt whose security response changed nothing (rotation
// disabled). An escalation that rotated templates is shown by TemplateRotationPanel instead. All wording comes from the
// backend; this component never decides retry or escalation.
export function AttemptAssessmentPanel({ result }: { result: AuthenticationDecision }) {
  const assessment = result.attempt_assessment
  if (!assessment || result.authentication_state !== 'ACCESS_DENIED') return null
  if (assessment.status === 'SUSPICIOUS_ATTEMPT' && result.template_rotation) return null
  const suspicious = assessment.status === 'SUSPICIOUS_ATTEMPT'
  const Icon = suspicious ? ShieldAlert : RotateCcw
  const modalities = ORDER.filter((m) => assessment.modality_messages[m])

  return (
    <div className={`mt-4 space-y-3 rounded-xl border p-5 ${suspicious ? 'border-warning/30 bg-warning/10' : 'border-primary/30 bg-primary/10'}`}>
      <div className="flex items-start gap-3">
        <Icon className={`mt-0.5 h-5 w-5 shrink-0 ${suspicious ? 'text-warning' : 'text-primary'}`} strokeWidth={1.5} />
        <div className="space-y-1 text-sm">
          <p className="font-medium text-foreground">{assessment.headline}</p>
          <p className="text-muted-foreground">{assessment.reason}</p>
          {ORDER.filter((m) => assessment.modality_hints?.[m]).map((m) => (
            <p key={m} className="text-xs text-muted-foreground">
              {MODALITY_LABEL[m]}: {assessment.modality_hints![m]}
            </p>
          ))}
          {assessment.attempt_notice && <p className="text-xs text-muted-foreground">{assessment.attempt_notice}</p>}
          <p className="text-muted-foreground">
            <span className="text-foreground">Security response:</span> {assessment.security_response}
          </p>
        </div>
      </div>
      {modalities.length > 1 && (
        <ul className="space-y-1 rounded-lg border border-border bg-card/60 px-4 py-3 text-sm">
          {modalities.map((m) => (
            <li key={m} className={assessment.modality_statuses[m] === 'VERIFIED' ? 'text-success' : 'text-foreground'}>
              <span className="text-muted-foreground">{MODALITY_LABEL[m]}:</span> {assessment.modality_messages[m]}
            </li>
          ))}
        </ul>
      )}
      {!suspicious && (
        <p className="text-xs text-muted-foreground">
          All selected factors are evaluated together in one attempt, so capture each of them again.
        </p>
      )}
    </div>
  )
}
