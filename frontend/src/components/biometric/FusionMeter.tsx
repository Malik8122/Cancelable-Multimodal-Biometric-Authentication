import { AnimatePresence, motion, useSpring, useTransform } from 'motion/react'
import { Check, Layers, X } from 'lucide-react'
import { useEffect } from 'react'
import type { Modality } from '../../api/types'
import { MODALITY_LABEL } from '../../config/buildings'
import { MODALITY_ICON } from '../../config/modalityIcons'

export type FusionOutcome = 'pending' | 'granted' | 'denied'

const R_ARC = 42
const R_CORE = 30
const GAP_DEG = 14

function arcPath(startDeg: number, endDeg: number, r: number) {
  const toXY = (deg: number) => {
    const rad = ((deg - 90) * Math.PI) / 180
    return [50 + r * Math.cos(rad), 50 + r * Math.sin(rad)]
  }
  const [x1, y1] = toXY(startDeg)
  const [x2, y2] = toXY(endDeg)
  const large = endDeg - startDeg > 180 ? 1 : 0
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`
}

/**
 * The live fusion meter shown while the backend verifies. One thin arc per SUBMITTED factor - it fills when that
 * factor's sample has been sent, it is NOT a confidence or a score. The core ring shows the request in flight and, only
 * when the backend's decision arrives, resolves (a spring) to green with a check or red with a cross.
 */
export function FusionMeter({ modalities, outcome }: { modalities: Modality[]; outcome: FusionOutcome }) {
  const n = Math.max(1, modalities.length)
  const span = 360 / n
  const resolved = useSpring(0, { stiffness: 120, damping: 20 })
  const resolvedOpacity = useTransform(resolved, [0, 0.2], [0, 1])
  useEffect(() => {
    resolved.set(outcome === 'pending' ? 0 : 1)
  }, [outcome, resolved])

  const coreClass = outcome === 'granted' ? 'stroke-success' : outcome === 'denied' ? 'stroke-danger' : 'stroke-processing'

  return (
    <div className="flex flex-col items-center">
      <div className="relative h-56 w-56 sm:h-64 sm:w-64">
        <svg viewBox="0 0 100 100" className="absolute inset-0 h-full w-full" aria-hidden>
          {modalities.map((m, i) => {
            const start = i * span + GAP_DEG / 2
            const end = (i + 1) * span - GAP_DEG / 2
            return (
              <g key={m}>
                <path d={arcPath(start, end, R_ARC)} fill="none" strokeWidth={1.5} className="stroke-border-strong" />
                <motion.path
                  d={arcPath(start, end, R_ARC)}
                  fill="none"
                  strokeWidth={2.5}
                  strokeLinecap="round"
                  className="stroke-info"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.6, delay: 0.15 + i * 0.18, ease: 'easeOut' }}
                />
              </g>
            )
          })}
          <circle cx="50" cy="50" r={R_CORE} fill="none" strokeWidth={1.5} className="stroke-border-strong" />
          {/* In flight: a short segment travels around the core (a loading indicator, stops on the decision). */}
          {outcome === 'pending' && (
            <motion.circle
              cx="50"
              cy="50"
              r={R_CORE}
              fill="none"
              strokeWidth={2.5}
              strokeLinecap="round"
              className="stroke-processing"
              style={{ pathLength: 0.22, rotate: 0 }}
              animate={{ rotate: 360 }}
              transition={{ duration: 1.4, repeat: Infinity, ease: 'linear' }}
            />
          )}
          <motion.circle
            cx="50"
            cy="50"
            r={R_CORE}
            fill="none"
            strokeWidth={3}
            strokeLinecap="round"
            className={coreClass}
            style={{ pathLength: resolved, opacity: resolvedOpacity, rotate: -90 }}
          />
        </svg>

        {/* Factor labels around the ring */}
        {modalities.map((m, i) => {
          const mid = ((i + 0.5) * span - 90) * (Math.PI / 180)
          const Icon = MODALITY_ICON[m]
          return (
            <motion.span
              key={m}
              className="absolute flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-info/40 bg-background text-info"
              style={{ left: `${50 + 50 * Math.cos(mid)}%`, top: `${50 + 50 * Math.sin(mid)}%` }}
              initial={{ opacity: 0, scale: 0.6 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 + i * 0.18 }}
              title={MODALITY_LABEL[m]}
            >
              <Icon className="h-4 w-4" strokeWidth={1.5} aria-hidden />
            </motion.span>
          )
        })}

        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={outcome}
              initial={{ opacity: 0, scale: 0.7 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.7 }}
              transition={{ type: 'spring', stiffness: 300, damping: 22 }}
              className={outcome === 'granted' ? 'text-success' : outcome === 'denied' ? 'text-danger' : 'text-processing'}
            >
              {outcome === 'granted' ? (
                <Check className="h-9 w-9" strokeWidth={2} aria-hidden />
              ) : outcome === 'denied' ? (
                <X className="h-9 w-9" strokeWidth={2} aria-hidden />
              ) : (
                <Layers className="h-8 w-8" strokeWidth={1.5} aria-hidden />
              )}
            </motion.span>
          </AnimatePresence>
          <span className="mt-1 text-xs font-semibold tracking-wide text-foreground uppercase">Fusion</span>
        </div>
      </div>
      <ul className="mt-2 flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {modalities.map((m) => (
          <li key={m} className="flex items-center gap-1.5">
            <Check className="h-3.5 w-3.5 text-info" strokeWidth={2} aria-hidden />
            {MODALITY_LABEL[m]} submitted
          </li>
        ))}
      </ul>
    </div>
  )
}
