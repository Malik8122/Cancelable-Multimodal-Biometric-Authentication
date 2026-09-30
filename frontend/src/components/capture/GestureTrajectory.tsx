import { motion } from 'motion/react'
import { cn } from '../../lib/utils'

/**
 * A captured hand trajectory (lib/biometricVisuals.ts - the real palm-centre path, fitted into 100 x 100). With `draw`
 * it traces itself once (SVG pathLength); otherwise it is shown complete, e.g. as an enrollment thumbnail.
 */
export function GestureTrajectory({
  path,
  draw,
  strokeWidth = 4,
  tone = 'success',
  label,
  onDrawn,
  className,
}: {
  path: string
  draw?: boolean
  strokeWidth?: number
  tone?: 'success' | 'primary'
  label: string
  onDrawn?: () => void
  className?: string
}) {
  const stroke = tone === 'success' ? 'stroke-success' : 'stroke-primary'
  return (
    <svg viewBox="0 0 100 100" role="img" aria-label={label} className={cn('h-full w-full overflow-visible', className)}>
      {draw && <path d={path} fill="none" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" className={cn(stroke, 'opacity-15')} />}
      <motion.path
        d={path}
        fill="none"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
        className={stroke}
        style={{ filter: draw ? 'drop-shadow(0 0 4px color-mix(in srgb, var(--color-success) 60%, transparent))' : undefined }}
        initial={draw ? { pathLength: 0 } : false}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.2, ease: 'easeInOut' }}
        onAnimationComplete={onDrawn}
      />
    </svg>
  )
}
