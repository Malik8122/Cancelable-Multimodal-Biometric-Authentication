import type { AuthenticationDecision, Modality, ModalityVerificationStatus } from '../../api/types'
import { MODALITY_LABEL } from '../../config/buildings'
import { FACTOR_ORDER, modalityOutcome } from '../../lib/outcomes'
import { DiagnosticsBox, type DiagnosticRow } from '../diagnostics/DiagnosticsBox'

const fmt = (v: number | undefined | null, d = 4) => (v === undefined || v === null ? '-' : v.toFixed(d))

const DECISION: Record<ModalityVerificationStatus, string> = {
  VERIFIED: 'VERIFIED',
  VERIFICATION_MISMATCH: 'NOT MATCHED (valid sample, compared)',
  QUALITY_INSUFFICIENT: 'QUALITY FAILURE (not compared)',
  CAPTURE_ERROR: 'CAPTURE ERROR (not compared)',
}

// Stage results for one presented factor, derived only from the backend's response.
function stages(result: AuthenticationDecision, m: Modality): { rows: DiagnosticRow[]; decision: string } {
  const status = modalityOutcome(result, m)
  const r = result.results?.[m]
  const captured = status !== 'CAPTURE_ERROR'
  const quality = captured && status !== 'QUALITY_INSUFFICIENT'
  const compared = quality && (!!r || (m === 'hand' && result.hand_gesture?.dtw_distance !== undefined))
  const hint = result.attempt_assessment?.modality_hints?.[m]
  const rows: DiagnosticRow[] = [
    { label: m === 'hand' ? 'Camera / sample decoded' : m === 'voice' ? 'Audio captured / decoded' : 'Image captured / face detected', value: captured ? '' : hint ?? '', status: captured ? 'pass' : 'fail' },
    { label: m === 'hand' ? 'Hand tracked / trajectory valid' : m === 'voice' ? 'Audio quality gate' : 'Capture quality', value: captured && !quality ? hint ?? '' : '', status: !captured ? 'info' : quality ? 'pass' : 'fail' },
    { label: 'Features / embedding + matching', value: compared ? 'performed' : 'not performed', status: compared ? 'pass' : 'info' },
  ]
  if (m === 'hand' && result.hand_gesture?.dtw_distance !== undefined) {
    const h = result.hand_gesture
    const perSample = h.dtw_distances ?? []
    rows.push(
      { label: 'Tracking rate', value: h.tracking_fps !== undefined ? `${h.tracking_fps} fps` : '-', status: h.tracking_fps === undefined ? undefined : h.tracking_fps >= 10 ? 'pass' : 'fail' },
      { label: 'Gesture duration', value: `${h.motion_duration_s ?? '-'} s` },
      { label: 'Z validity', value: h.z_validity ?? '-', status: h.z_validity === 'PASS' ? 'pass' : undefined },
      ...perSample.map((d, i) => ({ label: `DTW distance vs sample ${i + 1}`, value: d.toFixed(4) })),
      { label: `Distance (${h.aggregation ?? 'median'}) vs threshold`, value: `${fmt(h.dtw_distance)} <= ${fmt(h.threshold, 3)}`, status: h.decision === 'MATCH' ? 'pass' : 'fail' },
    )
  } else if (r) {
    const rule = r.metric_higher_is_better ? '>=' : '<='
    const name = r.metric === 'cosine_estimate' ? 'cosine similarity (estimate)' : r.metric === 'euclidean_estimate' ? 'Euclidean distance (estimate)' : r.metric
    rows.push(
      { label: `Raw metric: ${name}`, value: `${fmt(r.metric_value)} +/- ${fmt(r.metric_uncertainty, 3)} ${rule} ${fmt(r.metric_threshold, 3)}`, status: r.authenticated ? 'pass' : 'fail' },
      { label: 'Protected-template Hamming similarity', value: `${fmt(r.hamming_similarity)} (${r.template_bits} bits)` },
      { label: 'Fusion-scale score', value: fmt(r.score, 3) },
    )
  }
  return { rows, decision: DECISION[status] }
}

/**
 * Development diagnostics for the decision: per presented factor, which stage passed or failed, the raw metric with its
 * estimation uncertainty, the threshold and operator, and the decision - so a capture failure is never mistaken for a
 * mismatch. Present ONLY when the backend runs with DEBUG_SCORES=true (the response then carries per-modality results).
 */
export function ResearchDetails({ result }: { result: AuthenticationDecision }) {
  if (!result.results && !result.fusion_diagnostics && result.hand_gesture?.dtw_distance === undefined) return null
  const presented = FACTOR_ORDER.filter((m) => result.modalities_used.includes(m))
  return (
    <section className="space-y-3" aria-label="Developer diagnostics">
      {presented.map((m) => {
        const { rows, decision } = stages(result, m)
        return (
          <DiagnosticsBox
            key={m}
            title={`${MODALITY_LABEL[m]} - pipeline diagnostics`}
            rows={rows}
            footer={
              <>
                Decision: <span className="font-medium text-foreground">{decision}</span>
              </>
            }
          />
        )
      })}
      <DiagnosticsBox
        title="Fusion"
        rows={[
          { label: 'Modalities received', value: result.modalities_used.join(', ') || '-' },
          { label: 'Matched', value: result.matched_modalities.join(', ') || 'none' },
          { label: 'Policy', value: result.fusion_policy },
          { label: 'Fused score vs threshold', value: `${fmt(result.fusion_similarity, 3)} vs ${fmt(result.fusion_threshold, 3)}` },
        ]}
        footer={
          <>
            Decision: <span className="font-medium text-foreground">{result.authentication_state}</span>
          </>
        }
      />
    </section>
  )
}
