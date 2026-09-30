import type { AuthenticationDecision } from '../../api/types'
import { MODALITY_LABEL } from '../../config/buildings'

function Rows({ items }: { items: { label: string; value: string }[] }) {
  return (
    <dl className="divide-y divide-border rounded-xl border border-border bg-card/60 backdrop-blur-xl">
      {items.map((item) => (
        <div key={item.label} className="flex items-center justify-between px-6 py-3.5 text-sm">
          <dt className="text-muted-foreground">{item.label}</dt>
          <dd className="font-medium text-foreground">{item.value}</dd>
        </div>
      ))}
    </dl>
  )
}

const listModalities = (modalities: AuthenticationDecision['matched_modalities']) =>
  modalities.length ? modalities.map((m) => MODALITY_LABEL[m]).join(', ') : 'None'

// ACCESS_GRANTED: who was verified and where - the scores are in BiometricMetricsPanel.
// ACCESS_DENIED: every SUBMITTED modality was enrolled, but verification failed. No similarity is shown here - only the
// reason and which of the presented factors did verify.
export function DecisionSummary({ result, buildingName }: { result: AuthenticationDecision; buildingName: string }) {
  if (result.authentication_state === 'ACCESS_GRANTED') {
    return (
      <Rows
        items={[
          ...(result.display_name ? [{ label: 'User', value: result.display_name }] : []),
          { label: 'Building', value: buildingName },
        ]}
      />
    )
  }

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-danger/30 bg-danger/10 p-5 text-center">
        <p className="text-xs font-medium tracking-wide text-muted-foreground">Reason</p>
        <p className="mt-1 text-sm font-medium text-danger">
          {result.attempt_assessment?.reason ?? 'Biometric verification failed for one or more selected factors.'}
        </p>
      </div>
      <Rows
        items={[
          { label: 'Building', value: buildingName },
          { label: 'Presented Factors', value: listModalities(result.modalities_used) },
          { label: 'Verified Factors', value: listModalities(result.matched_modalities) },
        ]}
      />
    </div>
  )
}
