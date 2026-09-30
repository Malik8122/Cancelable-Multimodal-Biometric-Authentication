import { AnimatePresence, motion } from 'motion/react'
import { AlertTriangle, Check, Loader2, X } from 'lucide-react'
import { cn } from '../../lib/utils'
import { DURATION, EASE_OUT, spring } from '../../lib/motion'

/**
 * idle: nothing happening yet (waiting for a hand, camera ready). live: the sensor is actively tracking / listening.
 * processing: waiting for the backend. success / warning / danger: an outcome.
 */
export type StatusTone = 'idle' | 'live' | 'processing' | 'success' | 'warning' | 'danger'

const TONE: Record<StatusTone, string> = {
  idle: 'border-border-strong bg-raised/80 text-muted-foreground',
  live: 'border-info/40 bg-info/10 text-info',
  processing: 'border-primary/40 bg-primary/10 text-primary',
  success: 'border-success/40 bg-success/10 text-success',
  warning: 'border-warning/40 bg-warning/10 text-warning',
  danger: 'border-danger/40 bg-danger/10 text-danger',
}

// On top of a camera image the chip needs an opaque ground to stay readable.
const OVERLAY: Record<StatusTone, string> = {
  idle: 'border-white/15 bg-black/65 text-white/90',
  live: 'border-info/50 bg-black/70 text-info',
  processing: 'border-primary/50 bg-black/70 text-primary',
  success: 'border-success/50 bg-black/70 text-success',
  warning: 'border-warning/50 bg-black/70 text-warning',
  danger: 'border-danger/50 bg-black/70 text-danger',
}

function Indicator({ tone }: { tone: StatusTone }) {
  if (tone === 'processing') return <Loader2 className="h-3.5 w-3.5 animate-spin" strokeWidth={2} aria-hidden />
  if (tone === 'success') return <Check className="h-3.5 w-3.5" strokeWidth={2.25} aria-hidden />
  if (tone === 'warning') return <AlertTriangle className="h-3.5 w-3.5" strokeWidth={2} aria-hidden />
  if (tone === 'danger') return <X className="h-3.5 w-3.5" strokeWidth={2.25} aria-hidden />
  return (
    <span aria-hidden className="relative flex h-2 w-2">
      {/* The ping only runs while a sensor is live - it says "actively tracking", nothing else loops. */}
      {tone === 'live' && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-60" />}
      <span className="relative inline-flex h-2 w-2 rounded-full bg-current" />
    </span>
  )
}

/**
 * The single status chip every capture uses. On a state change the chip morphs: its width springs to the new label
 * (layout animation) while the label and icon cross-fade - it never jumps. It is announced to screen readers
 * (aria-live), and the meaning is always in the text and the icon, never only in the color.
 */
export function CaptureStatus({ tone, label, overlay, className }: { tone: StatusTone; label: string; overlay?: boolean; className?: string }) {
  return (
    <motion.span
      layout
      transition={spring}
      role="status"
      aria-live="polite"
      className={cn(
        'inline-flex max-w-full items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium transition-colors duration-200',
        overlay ? OVERLAY[tone] : TONE[tone],
        overlay && 'backdrop-blur-sm',
        className,
      )}
    >
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.span
          key={tone}
          layout="position"
          className="flex"
          initial={{ opacity: 0, scale: 0.6 }}
          animate={{ opacity: 1, scale: 1, transition: { duration: DURATION.base, ease: EASE_OUT } }}
          exit={{ opacity: 0, scale: 0.6, transition: { duration: DURATION.fast } }}
        >
          <Indicator tone={tone} />
        </motion.span>
      </AnimatePresence>
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.span
          key={label}
          layout="position"
          className="truncate"
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0, transition: { duration: DURATION.base, ease: EASE_OUT } }}
          exit={{ opacity: 0, y: -4, transition: { duration: DURATION.fast } }}
        >
          {label}
        </motion.span>
      </AnimatePresence>
    </motion.span>
  )
}
