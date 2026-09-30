import { motion } from 'motion/react'
import { RefreshCw, ShieldAlert } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { AuthenticationDecision, Modality } from '../../api/types'
import { MODALITY_LABEL } from '../../config/buildings'

const ORDER: Modality[] = ['face', 'voice', 'hand']

// Shown only when the BACKEND escalated a SUSPICIOUS_ATTEMPT (repeated high-quality mismatches) and reports a template
// rotation - never after a single failed verification (the client never decides either). The headline, reason and
// security response are the backend's own wording: a suspicious verification pattern, not a detected attacker.
// The rotation made visible: the active credential card turns over from the old set to the new one. Shown ONLY for a
// rotation the backend reports as ROTATED - otherwise it stays on the old set.
function RotationFlip({ before, after, rotated }: { before: number; after: number; rotated: boolean }) {
  const [flipped, setFlipped] = useState(false)
  useEffect(() => {
    if (!rotated) return
    const t = window.setTimeout(() => setFlipped(true), 700)
    return () => window.clearTimeout(t)
  }, [rotated])
  const face = 'absolute inset-0 flex flex-col items-center justify-center rounded-2xl border [backface-visibility:hidden]'
  return (
    <div className="flex items-center justify-center gap-4 py-2" style={{ perspective: 800 }}>
      <motion.div
        className="relative h-28 w-24 [transform-style:preserve-3d]"
        animate={{ rotateY: flipped ? 180 : 0 }}
        transition={{ type: 'spring', stiffness: 120, damping: 16 }}
        aria-label={rotated ? `Active template set rotated from T${before} to T${after}` : `Active template set T${before}`}
        role="img"
      >
        <div className={`${face} border-danger/50 bg-danger/10`}>
          <span className="font-mono text-2xl font-semibold text-foreground">T{before}</span>
          <span className="mt-1 text-[11px] font-medium text-danger">{rotated ? 'Revoked' : 'Active'}</span>
        </div>
        <div className={`${face} border-success/50 bg-success/10 [transform:rotateY(180deg)]`}>
          <span className="font-mono text-2xl font-semibold text-foreground">T{after}</span>
          <span className="mt-1 text-[11px] font-medium text-success">Now active</span>
        </div>
      </motion.div>
    </div>
  )
}

export function TemplateRotationPanel({ result }: { result: AuthenticationDecision }) {
  const rotation = result.template_rotation
  const assessment = result.attempt_assessment
  if (!rotation || !assessment || assessment.status !== 'SUSPICIOUS_ATTEMPT' || result.authentication_state !== 'ACCESS_DENIED') {
    return null
  }
  const rotated = rotation.status === 'ROTATED'
  const modalities = ORDER.filter((m) => rotation.modalities.includes(m))

  return (
    <div className="mt-4 space-y-4 rounded-2xl border border-warning/30 bg-warning/10 p-5">
      <RotationFlip before={rotation.active_set_before} after={rotation.active_set_after} rotated={rotated} />
      <div className="flex items-start gap-3">
        <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-warning" strokeWidth={1.5} />
        <div className="space-y-1 text-sm">
          <p className="font-medium text-foreground">
            {assessment.headline}
          </p>
          <p className="text-muted-foreground">
            <span className="text-foreground">Reason:</span> {assessment.reason}
          </p>
          {assessment.attempt_notice && <p className="text-xs text-muted-foreground">{assessment.attempt_notice}</p>}
          <p className="text-muted-foreground">
            <span className="text-foreground">Security response:</span>{' '}
            {assessment.security_response}
          </p>
          <p className="text-muted-foreground">
            <span className="text-foreground">Template version:</span>{' '}
            <span className="font-mono text-foreground">
              {rotation.active_set_before} &rarr; {rotation.active_set_after}
            </span>
          </p>
        </div>
      </div>

      <div className="overflow-hidden rounded-lg border border-border bg-card/60">
        <p className="border-b border-border px-4 py-2 text-xs font-medium tracking-wide text-muted-foreground">
          MODALITY TEMPLATE STATUS
        </p>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-muted-foreground">
              <th className="px-4 py-2 text-left font-medium">Modality</th>
              <th className="px-4 py-2 text-center font-medium">Before</th>
              <th className="px-4 py-2 text-center font-medium">After</th>
            </tr>
          </thead>
          <tbody>
            {modalities.map((m) => (
              <tr key={m} className="border-t border-border">
                <td className="px-4 py-2 text-foreground">{MODALITY_LABEL[m]}</td>
                <td className="px-4 py-2 text-center font-mono text-muted-foreground">T{rotation.active_set_before}</td>
                <td className={`px-4 py-2 text-center font-mono ${rotated ? 'font-semibold text-warning' : 'text-muted-foreground'}`}>
                  T{rotation.active_set_after}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {rotated && (
        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          <RefreshCw className="h-3.5 w-3.5 shrink-0" strokeWidth={1.5} />
          Templates rotated after a suspicious authentication attempt. The new active set was generated at enrollment from the
          registered user&apos;s own biometrics under new keys - the failed sample was not stored. {rotation.remaining_standby_sets}{' '}
          standby set{rotation.remaining_standby_sets === 1 ? '' : 's'} left.
        </p>
      )}
    </div>
  )
}
