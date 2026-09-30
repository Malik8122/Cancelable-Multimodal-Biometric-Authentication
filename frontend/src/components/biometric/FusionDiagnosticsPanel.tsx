import { useState } from 'react'
import { ArrowDown, ChevronDown, ChevronUp } from 'lucide-react'
import type { AuthenticationDecision, FusionModalityDiagnostics, FusionModalityStatus } from '../../api/types'
import { MODALITY_LABEL } from '../../config/buildings'

const MODALITY_ORDER: ('face' | 'voice' | 'hand')[] = ['face', 'voice', 'hand']

const STATUS_LABEL: Record<FusionModalityStatus, string> = {
  verified: 'Verified',
  failed_below_threshold: 'Failed',
  not_presented: 'Not presented',
  capture_error: 'Capture error',
}

const STATUS_CLASS: Record<FusionModalityStatus, string> = {
  verified: 'text-success',
  failed_below_threshold: 'text-danger',
  not_presented: 'text-muted-foreground',
  capture_error: 'text-warning',
}

const POLICY_LABEL: Record<string, string> = {
  ALL_REQUIRED: 'All Required',
  AT_LEAST_TWO: 'At Least Two',
  WEIGHTED: 'Weighted',
}

function ModalityCard({ label, data }: { label: string; data: FusionModalityDiagnostics }) {
  return (
    <div className="rounded-lg border border-border bg-background/40 px-4 py-3">
      <p className="text-xs font-medium text-muted-foreground">{label}</p>
      <p className="mt-1.5 font-mono text-sm text-foreground">Score: {data.score === null ? '—' : data.score.toFixed(4)}</p>
      <p className={`mt-0.5 text-xs font-medium ${STATUS_CLASS[data.status]}`}>Status: {STATUS_LABEL[data.status]}</p>
    </div>
  )
}

// Local-development / diagnostic only: exactly the values the real backend fusion engine
// (fusion/score_fusion.py + fusion/policy.py, via fusion/diagnostics.py) already produced for
// this request - nothing here is recomputed client-side. Renders only when the backend ran with
// DEBUG_SCORES=true (result.fusion_diagnostics present); silently absent otherwise, same as every
// other debug-only field on this response.
export function FusionDiagnosticsPanel({ result }: { result: AuthenticationDecision }) {
  const [open, setOpen] = useState(false)
  const diagnostics = result.fusion_diagnostics
  if (!diagnostics) return null

  const granted = diagnostics.access_granted
  const fusedAboveThreshold = diagnostics.fused_score >= diagnostics.threshold
  // The fused number alone can clear (or miss) the threshold while the actual policy decides
  // differently (e.g. ALL_REQUIRED denies on a single failed factor regardless of the average) -
  // surface that mismatch explicitly rather than letting the fused score look authoritative.
  const policyOverrodeFusedScore = fusedAboveThreshold !== granted
  const policyLabel = POLICY_LABEL[diagnostics.policy] ?? diagnostics.policy

  return (
    <div className="mb-8 overflow-hidden rounded-xl border border-border bg-card/60 backdrop-blur-xl">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-6 py-3.5 text-left text-sm font-medium text-foreground"
      >
        <span className="flex items-center gap-2">
          Fusion Diagnostics
          <span className="rounded-full border border-border px-2 py-0.5 text-[10px] font-normal text-muted-foreground">
            Local development only
          </span>
        </span>
        {open ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" strokeWidth={1.5} />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground" strokeWidth={1.5} />
        )}
      </button>

      {open && (
        <div className="border-t border-border px-6 py-5">
          <p className="mb-4 flex items-center justify-center gap-1.5 text-center text-[11px] font-medium tracking-wide text-muted-foreground">
            MODALITY SCORES <ArrowDown className="h-3 w-3" strokeWidth={2} /> FUSION <ArrowDown className="h-3 w-3" strokeWidth={2} /> FINAL DECISION
          </p>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {MODALITY_ORDER.map((modality) => {
              const data = diagnostics[modality]
              return data ? <ModalityCard key={modality} label={MODALITY_LABEL[modality]} data={data} /> : null
            })}
          </div>

          <div className="my-3 flex justify-center">
            <ArrowDown className="h-4 w-4 text-muted-foreground" strokeWidth={1.5} />
          </div>

          <div className="rounded-lg border border-border bg-background/40 px-4 py-3.5">
            <p className="mb-2 text-xs font-medium text-muted-foreground">Fusion</p>
            <dl className="space-y-1.5 text-sm">
              <div className="flex items-center justify-between">
                <dt className="text-muted-foreground">Fused Score</dt>
                <dd className="font-mono text-foreground">{diagnostics.fused_score.toFixed(4)}</dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-muted-foreground">Threshold</dt>
                <dd className="font-mono text-foreground">{diagnostics.threshold.toFixed(4)}</dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-muted-foreground">Policy</dt>
                <dd className="text-foreground">{policyLabel}</dd>
              </div>
            </dl>
          </div>

          <div className="my-3 flex justify-center">
            <ArrowDown className="h-4 w-4 text-muted-foreground" strokeWidth={1.5} />
          </div>

          <div className={`rounded-lg border px-4 py-3.5 text-center ${granted ? 'border-success/30 bg-success/10' : 'border-danger/30 bg-danger/10'}`}>
            <p className="text-xs font-medium text-muted-foreground">Final Decision</p>
            <p className={`mt-1 text-lg font-semibold tracking-tight ${granted ? 'text-success' : 'text-danger'}`}>
              {granted ? 'ACCESS GRANTED' : 'ACCESS DENIED'}
            </p>
            {policyOverrodeFusedScore && (
              <p className="mx-auto mt-2 max-w-sm text-[11px] text-muted-foreground">
                {fusedAboveThreshold
                  ? `The fused score alone was above threshold, but the ${policyLabel} policy requires every presented factor to individually verify - it does not decide on the fused score alone.`
                  : `The fused score alone was below threshold, but the ${policyLabel} policy's own per-factor evaluation produced a different result.`}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
