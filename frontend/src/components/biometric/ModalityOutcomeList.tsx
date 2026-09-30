import { motion } from 'motion/react'
import { AlertTriangle, Check, X } from 'lucide-react'
import type { AuthenticationDecision, ModalityVerificationStatus } from '../../api/types'
import { FACTOR_ORDER, modalityOutcome } from '../../lib/outcomes'
import { MODALITY_LABEL } from '../../config/buildings'
import { cn } from '../../lib/utils'
import { rise, stagger } from '../../lib/motion'
import { ModalityBadge } from './ModalityBadge'

const LABEL: Record<ModalityVerificationStatus, string> = {
  VERIFIED: 'Verified',
  VERIFICATION_MISMATCH: 'Not matched',
  QUALITY_INSUFFICIENT: 'Capture quality too low',
  CAPTURE_ERROR: 'Capture not usable',
}

export function ModalityOutcomeList({ result }: { result: AuthenticationDecision }) {
  const presented = FACTOR_ORDER.filter((m) => result.modalities_used.includes(m))
  const hints = result.attempt_assessment?.modality_hints ?? {}

  return (
    <motion.ul variants={stagger} initial="hidden" animate="show" className="divide-y divide-border">
      {presented.map((m) => {
        const status = modalityOutcome(result, m)
        const ok = status === 'VERIFIED'
        const capture = status === 'QUALITY_INSUFFICIENT' || status === 'CAPTURE_ERROR'
        const hint = hints[m] ?? (m === 'hand' && capture ? result.hand_gesture?.reason : undefined)
        return (
          <motion.li key={m} variants={rise} className="flex items-center gap-3 px-4 py-3 sm:px-5">
            <ModalityBadge modality={m} size="sm" tone={ok ? 'success' : capture ? 'warning' : 'danger'} />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium text-foreground">{MODALITY_LABEL[m]}</p>
              {hint && <p className="text-meta text-muted-foreground">{hint}</p>}
            </div>
            <span className={cn('flex shrink-0 items-center gap-1.5 text-sm font-medium', ok ? 'text-success' : capture ? 'text-warning' : 'text-danger')}>
              {ok ? <Check className="h-4 w-4" strokeWidth={2.25} aria-hidden /> : capture ? <AlertTriangle className="h-4 w-4" strokeWidth={2} aria-hidden /> : <X className="h-4 w-4" strokeWidth={2.25} aria-hidden />}
              {LABEL[status]}
            </span>
          </motion.li>
        )
      })}
    </motion.ul>
  )
}
