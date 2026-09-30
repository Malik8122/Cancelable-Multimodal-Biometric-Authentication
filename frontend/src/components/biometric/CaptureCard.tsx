import { motion } from 'motion/react'
import type { ReactNode } from 'react'
import type { Modality } from '../../api/types'
import { cn } from '../../lib/utils'
import { ModalityBadge } from './ModalityBadge'

export type CaptureCardState = 'idle' | 'active' | 'complete' | 'attention'

const BORDER: Record<CaptureCardState, string> = {
  idle: '',
  active: 'border-primary/45',
  complete: 'border-success/40',
  attention: 'border-warning/45',
}

/**
 * The shared shell of every biometric capture (face, voice sentence, hand gesture): the modality, what this capture
 * is for (eyebrow: "VOICE ENROLLMENT" / "VOICE VERIFICATION"), its title, and an optional right-hand slot for the
 * progress ("Sentence 1 / 2", "Gesture 3 / 5"). The border tells the card's state at a glance; the text says it too.
 */
export function CaptureCard({
  modality,
  eyebrow,
  title,
  aside,
  state = 'idle',
  children,
  className,
}: {
  modality: Modality
  eyebrow: string
  title: string
  aside?: ReactNode
  state?: CaptureCardState
  children: ReactNode
  className?: string
}) {
  return (
    <motion.section
      layout="position"
      aria-label={`${eyebrow}: ${title}`}
      className={cn('surface-card p-4 transition-colors duration-300 sm:p-5', BORDER[state], className)}
    >
      <header className="mb-4 flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <ModalityBadge modality={modality} tone={state === 'complete' ? 'success' : state === 'active' ? 'active' : 'default'} />
          <div className="min-w-0">
            <p className="text-eyebrow text-muted-foreground">{eyebrow}</p>
            <h3 className="text-card-title truncate text-foreground">{title}</h3>
          </div>
        </div>
        {aside && <div className="shrink-0 pt-0.5">{aside}</div>}
      </header>
      {children}
    </motion.section>
  )
}

/** A compact "3 / 5" counter for the header's aside slot. */
export function StepCount({ current, total, label }: { current: number; total: number; label: string }) {
  return (
    <span className="rounded-full border border-border-strong bg-raised/70 px-2.5 py-1 text-xs font-medium tabular-nums text-foreground">
      <span className="sr-only">{label} </span>
      {current} / {total}
    </span>
  )
}
