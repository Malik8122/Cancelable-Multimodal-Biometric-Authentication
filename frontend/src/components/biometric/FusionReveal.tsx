import { AnimatePresence, LayoutGroup, motion } from 'motion/react'
import { Layers } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { AuthenticationDecision, Modality } from '../../api/types'
import { MODALITY_LABEL } from '../../config/buildings'
import { MODALITY_ICON } from '../../config/modalityIcons'
import { useReducedMotion } from '../../hooks/useReducedMotion'
import { FACTOR_ORDER, modalityOutcome, outcomeTone, type OutcomeTone } from '../../lib/outcomes'
import { cn } from '../../lib/utils'
import { AccessDecisionHero } from './AccessDecisionHero'

type Stage = 'scan' | 'converge' | 'verdict'

const SCAN_MS = 1300
const CONVERGE_MS = 900

// The Z gesture emblem (a symbol of the modality - not the user's trajectory).
const Z_EMBLEM = 'M30 9 L70 9 L30 41 L70 41'

// Stylised emblem of each factor with its ~1 s scan treatment. These are symbols of the modality, not captured data.
function Emblem({ modality }: { modality: Modality }) {
  const Icon = MODALITY_ICON[modality]
  if (modality === 'face') {
    return (
      <div className="relative flex h-full w-full items-center justify-center">
        {['top-0 left-0 border-t-2 border-l-2', 'top-0 right-0 border-t-2 border-r-2', 'bottom-0 left-0 border-b-2 border-l-2', 'bottom-0 right-0 border-b-2 border-r-2'].map((c) => (
          <span key={c} className={`absolute h-4 w-4 rounded-[3px] border-info ${c}`} />
        ))}
        <Icon className="h-8 w-8 text-info" strokeWidth={1.25} />
        <motion.span
          className="absolute inset-x-1 h-0.5 rounded-full bg-info shadow-[0_0_8px_var(--color-info)]"
          initial={{ top: '8%' }}
          animate={{ top: '88%' }}
          transition={{ duration: 1, ease: 'easeInOut' }}
        />
      </div>
    )
  }
  if (modality === 'voice') {
    const bars = [0.35, 0.6, 0.9, 0.5, 1, 0.7, 0.4, 0.8, 0.55, 0.3]
    return (
      <div className="flex h-full w-full items-center justify-center gap-[3px]">
        {bars.map((h, i) => (
          <motion.span
            key={i}
            className="w-1 rounded-full bg-info"
            style={{ height: `${h * 60}%` }}
            initial={{ opacity: 0.25 }}
            animate={{ opacity: [0.25, 1, 0.55] }}
            transition={{ duration: 0.5, delay: i * 0.05, ease: 'easeOut' }}
          />
        ))}
      </div>
    )
  }
  return (
    <svg viewBox="0 0 100 50" className="h-full w-full" aria-hidden>
      <path d={Z_EMBLEM} fill="none" strokeWidth={2} strokeLinejoin="round" className="stroke-info/20" />
      <motion.path d={Z_EMBLEM} fill="none" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" className="stroke-info" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1, ease: 'easeInOut' }} />
    </svg>
  )
}

const TONE_RING: Record<OutcomeTone, string> = {
  success: 'border-success/60 bg-success/10 text-success',
  warning: 'border-warning/60 bg-warning/10 text-warning',
  danger: 'border-danger/60 bg-danger/10 text-danger',
}

/**
 * Signature moment: the verdict is not shown at once. Each presented factor gets a short scan treatment, the factors
 * then converge (shared-layout animation) toward the centre - each already tinted with ITS real outcome - and merge
 * into the shield, which states the decision. Everything shown is the backend's response: which factors were
 * presented, how each one ended, and the final decision. Reduced motion (or Skip) goes straight to the verdict.
 */
export function FusionReveal({ result, onRevealed }: { result: AuthenticationDecision; onRevealed: () => void }) {
  const reducedMotion = useReducedMotion()
  const [stage, setStage] = useState<Stage>(reducedMotion ? 'verdict' : 'scan')
  const presented = FACTOR_ORDER.filter((m) => result.modalities_used.includes(m))
  const granted = result.authentication_state === 'ACCESS_GRANTED'
  // Held in a ref so a parent re-render (new callback) never restarts the stage timer.
  const onRevealedRef = useRef(onRevealed)
  useEffect(() => {
    onRevealedRef.current = onRevealed
  }, [onRevealed])

  useEffect(() => {
    if (stage === 'verdict') {
      onRevealedRef.current()
      return
    }
    const t = window.setTimeout(() => setStage(stage === 'scan' ? 'converge' : 'verdict'), stage === 'scan' ? SCAN_MS : CONVERGE_MS)
    return () => window.clearTimeout(t)
  }, [stage])

  return (
    <div className="relative">
      <LayoutGroup>
        <AnimatePresence mode="wait">
          {stage !== 'verdict' ? (
            <motion.div key="journey" exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.2 } }} className="flex min-h-[19rem] flex-col items-center justify-center py-6">
              <p className="text-eyebrow mb-1 text-muted-foreground">Verification replay</p>
              <p className="text-section-title mb-8 text-foreground" aria-live="polite">
                {stage === 'scan' ? 'Presented biometric factors' : 'Multimodal fusion'}
              </p>
              {stage === 'scan' ? (
                <div className="flex flex-wrap items-start justify-center gap-4 sm:gap-8">
                  {presented.map((m) => (
                    <div key={m} className="flex flex-col items-center gap-2">
                      <motion.div layoutId={`reveal-${m}`} className="h-20 w-20 rounded-2xl border border-info/40 bg-info/[0.07] p-3 sm:h-24 sm:w-24" style={{ borderRadius: 16 }}>
                        <Emblem modality={m} />
                      </motion.div>
                      <span className="text-xs font-medium text-foreground">{MODALITY_LABEL[m]}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="relative h-40 w-40">
                  {/* The factors gather around the fusion core, each tinted by its real outcome. */}
                  <motion.div
                    className="absolute inset-7 flex flex-col items-center justify-center rounded-full border border-processing/50 bg-processing/10 text-processing"
                    initial={{ scale: 0.4, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.4 }}
                  >
                    <Layers className="h-6 w-6" strokeWidth={1.5} aria-hidden />
                    <span className="mt-1 text-[11px] font-semibold tracking-wide text-foreground uppercase">Fusion</span>
                  </motion.div>
                  {presented.map((m, i) => {
                    const angle = ((i / presented.length) * 360 - 90) * (Math.PI / 180)
                    const Icon = MODALITY_ICON[m]
                    const tone = outcomeTone(modalityOutcome(result, m))
                    return (
                      <motion.div
                        key={m}
                        layoutId={`reveal-${m}`}
                        transition={{ type: 'spring', stiffness: 170, damping: 22 }}
                        className={cn('absolute flex h-11 w-11 -translate-x-1/2 -translate-y-1/2 items-center justify-center border', TONE_RING[tone])}
                        style={{ borderRadius: 999, left: `${50 + 42 * Math.cos(angle)}%`, top: `${50 + 42 * Math.sin(angle)}%` }}
                      >
                        <Icon className="h-5 w-5" strokeWidth={1.5} />
                      </motion.div>
                    )
                  })}
                </div>
              )}
              <button type="button" onClick={() => setStage('verdict')} className="btn btn-ghost mt-8 min-h-9 px-3 text-xs">
                Skip to decision
              </button>
            </motion.div>
          ) : (
            <motion.div key="verdict" initial={{ opacity: 0, scale: 0.94 }} animate={{ opacity: 1, scale: 1 }} transition={{ type: 'spring', stiffness: 220, damping: 24 }}>
              <AccessDecisionHero
                authenticated={granted}
                subtitle={
                  granted
                    ? `Multimodal authentication successful${result.display_name ? ` - welcome, ${result.display_name}` : ''}.`
                    : 'One or more biometric checks did not satisfy the configured verification criteria.'
                }
              />
            </motion.div>
          )}
        </AnimatePresence>
      </LayoutGroup>
    </div>
  )
}
