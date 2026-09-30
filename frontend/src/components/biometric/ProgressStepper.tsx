import { motion } from 'motion/react'
import { AlertTriangle, Check } from 'lucide-react'
import type { ReactNode } from 'react'
import { cn } from '../../lib/utils'
import { DURATION, EASE_OUT } from '../../lib/motion'

export type StepState = 'done' | 'current' | 'upcoming' | 'attention'

export interface Step {
  key: string
  label: string
  /** Short secondary line: "Complete", "2 / 5 gestures", "Optional". */
  detail?: string
  state: StepState
  icon?: ReactNode
}

const STATE_TEXT: Record<StepState, string> = { done: 'complete', current: 'current step', upcoming: 'not started', attention: 'needs attention' }

function Marker({ step, index }: { step: Step; index: number }) {
  const base = 'relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border text-xs font-semibold transition-colors duration-300'
  if (step.state === 'done')
    return (
      <span className={cn(base, 'border-success/50 bg-success/15 text-success')}>
        <motion.span initial={{ scale: 0.4, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: DURATION.base, ease: EASE_OUT }}>
          <Check className="h-4 w-4" strokeWidth={2.25} aria-hidden />
        </motion.span>
      </span>
    )
  if (step.state === 'attention')
    return (
      <span className={cn(base, 'border-warning/50 bg-warning/15 text-warning')}>
        <AlertTriangle className="h-4 w-4" strokeWidth={2} aria-hidden />
      </span>
    )
  return (
    <span className={cn(base, step.state === 'current' ? 'border-primary bg-primary/15 text-primary' : 'border-border-strong bg-raised text-muted-foreground')}>
      {step.icon ?? index + 1}
    </span>
  )
}

/**
 * A compact horizontal progress indicator (detail lines appear from the small breakpoint up). The connector after a finished step
 * fills in, so progress is visible as movement; the state is also in each marker's icon and in screen-reader text.
 */
export function ProgressStepper({ steps, label, className }: { steps: Step[]; label: string; className?: string }) {
  return (
    <nav aria-label={label} className={className}>
      <ol className="flex items-start">
        {steps.map((step, i) => {
          const last = i === steps.length - 1
          return (
            <li key={step.key} aria-current={step.state === 'current' ? 'step' : undefined} className="relative flex min-w-0 flex-1 flex-col items-center gap-1.5 text-center sm:gap-2">
              {!last && (
                <span aria-hidden className="absolute top-4 left-[calc(50%+1.25rem)] block h-px w-[calc(100%-2.5rem)] bg-border-strong">
                  <motion.span
                    className="block h-full origin-left bg-success/70"
                    initial={false}
                    animate={{ scaleX: step.state === 'done' ? 1 : 0 }}
                    transition={{ duration: DURATION.slow, ease: EASE_OUT }}
                  />
                </span>
              )}
              <Marker step={step} index={i} />
              <div className="min-w-0">
                <p className={cn('text-xs leading-tight font-medium sm:text-sm', step.state === 'upcoming' ? 'text-muted-foreground' : 'text-foreground')}>{step.label}</p>
                {step.detail && <p className="text-meta hidden text-muted-foreground sm:block">{step.detail}</p>}
                <span className="sr-only">({STATE_TEXT[step.state]})</span>
              </div>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
