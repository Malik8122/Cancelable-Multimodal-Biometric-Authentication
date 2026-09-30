import { Check, X } from 'lucide-react'
import type { AuthenticationDecision, Modality } from '../../api/types'
import { MODALITY_LABEL } from '../../config/buildings'

const ORDER: Modality[] = ['face', 'voice', 'hand']

// Scores are shown as percentages of the backend's higher-is-better fusion-scale score (the same value fusion uses).
// For voice and hand this is converted from a distance - never the raw distance - and each factor's raw metric is
// different (cosine estimate, Euclidean estimate, DTW distance), so the per-factor numbers are NOT directly comparable.
// Negative scores (unrelated samples) are shown as 0%.
const percent = (score: number) => `${(Math.min(1, Math.max(0, score)) * 100).toFixed(1)}%`

function Row({ label, score, pass }: { label: string; score: number; pass: boolean }) {
  return (
    <div className="flex items-center justify-between px-4 py-3 text-sm sm:px-5">
      <span className="text-muted-foreground">{label}</span>
      <span className={`flex items-center gap-2 font-medium ${pass ? 'text-success' : 'text-danger'}`}>
        <span className="font-mono tabular-nums text-foreground">{percent(score)}</span>
        {pass ? <Check className="h-4 w-4" strokeWidth={2} aria-label="pass" /> : <X className="h-4 w-4" strokeWidth={2} aria-label="fail" />}
      </span>
    </div>
  )
}

// The fusion score (on success) and - only when the backend runs with DEBUG_SCORES=true - one fusion-scale score per
// presented factor. The decision itself is shown by AccessDecisionHero and ModalityOutcomeList, not here.
export function BiometricMetricsPanel({ result }: { result: AuthenticationDecision }) {
  const granted = result.authentication_state === 'ACCESS_GRANTED'
  const results = result.results ?? {}
  const present = ORDER.filter((m) => results[m])
  if (present.length === 0 && !granted) return null

  return (
    <section className="surface-card mb-4 overflow-hidden" aria-labelledby="scores-title">
      <div className="border-b border-border px-4 py-3 sm:px-5">
        <h2 id="scores-title" className="text-card-title text-foreground">
          {present.length ? 'Scores (debug view)' : 'Fusion score'}
        </h2>
        <p className="text-meta text-muted-foreground">
          {present.length
            ? 'On the common fusion scale, higher is better. Each factor uses a different underlying metric, so per-factor numbers are not directly comparable.'
            : `Mean of the presented factors' fusion-scale scores; threshold ${percent(result.fusion_threshold)}.`}
        </p>
      </div>
      <div className="divide-y divide-border">
        {present.map((m) => (
          <Row key={m} label={MODALITY_LABEL[m]} score={results[m]!.score} pass={results[m]!.authenticated} />
        ))}
        <Row label="Fusion score" score={result.fusion_similarity} pass={result.fusion_similarity >= result.fusion_threshold} />
      </div>
    </section>
  )
}
