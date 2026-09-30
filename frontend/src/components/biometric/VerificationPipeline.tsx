import { motion } from 'motion/react'
import { Check, Layers, Loader2, ShieldCheck } from 'lucide-react'
import type { ReactNode } from 'react'
import type { Modality } from '../../api/types'
import { MODALITY_LABEL } from '../../config/buildings'
import { cn } from '../../lib/utils'
import { DURATION, EASE_OUT } from '../../lib/motion'
import { MODALITY_ICON } from '../../config/modalityIcons'

export type PipelinePhase = 'capture' | 'processing' | 'done'

/**
 * The authentication flow as one picture: the presented factors (Face / Voice / Hand Gesture) converge into the fusion
 * engine, which produces one access decision. During capture each factor shows whether it has been captured; while the
 * backend verifies, the connectors carry a moving pulse into fusion. The backend verifies everything in ONE request,
 * so no per-factor verdict is shown here before the response arrives - the verdicts are on the result page.
 */
export function VerificationPipeline({
  modalities,
  captured,
  phase,
  className,
}: {
  modalities: Modality[]
  captured: Partial<Record<Modality, boolean>>
  phase: PipelinePhase
  className?: string
}) {
  const n = Math.max(modalities.length, 1)
  const allCaptured = modalities.length > 0 && modalities.every((m) => captured[m])
  const flowing = phase === 'processing'

  const fusionLabel = phase === 'capture' ? (allCaptured ? 'Ready' : 'Awaiting captures') : phase === 'processing' ? 'Fusing' : 'Complete'
  const decisionLabel = phase === 'done' ? 'Ready' : 'Pending'

  return (
    <div className={cn('surface-card px-3 py-4 sm:px-5', className)}>
      <p className="sr-only" role="status" aria-live="polite">
        {phase === 'capture'
          ? `${modalities.filter((m) => captured[m]).length} of ${modalities.length} factors captured.`
          : phase === 'processing'
            ? 'Running multimodal verification.'
            : 'Verification complete.'}
      </p>
      <div aria-hidden className="grid grid-cols-[auto_minmax(1.5rem,1fr)_auto_minmax(1rem,2.5rem)_auto] items-center">
        {/* Factors */}
        <div className="flex flex-col gap-2">
          {modalities.map((m) => {
            const Icon = MODALITY_ICON[m]
            const isCaptured = !!captured[m]
            return (
              <motion.div
                key={m}
                layout
                className={cn(
                  'flex items-center gap-2 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors duration-300',
                  isCaptured ? 'border-success/40 bg-success/10 text-foreground' : 'border-dashed border-border-strong bg-raised/50 text-muted-foreground',
                )}
              >
                <Icon className={cn('h-4 w-4', isCaptured ? 'text-success' : 'text-muted-foreground')} strokeWidth={1.5} />
                <span className="hidden sm:inline">{MODALITY_LABEL[m]}</span>
                {isCaptured ? (
                  <motion.span initial={{ scale: 0.4, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: DURATION.base, ease: EASE_OUT }}>
                    <Check className="h-3.5 w-3.5 text-success" strokeWidth={2.25} />
                  </motion.span>
                ) : (
                  <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground/60" />
                )}
              </motion.div>
            )
          })}
        </div>

        {/* Converging connectors - absolutely positioned so the SVG takes the row's height, not its own. */}
        <div className="relative h-full min-h-10 self-stretch">
        <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none">
          {modalities.map((m, i) => {
            const y = ((i + 0.5) / n) * 100
            const d = `M0 ${y} C 55 ${y}, 45 50, 100 50`
            return (
              <g key={m}>
                <path d={d} fill="none" vectorEffect="non-scaling-stroke" strokeWidth={1.25} className={captured[m] ? 'stroke-success/50' : 'stroke-border-strong'} />
                {flowing && (
                  <motion.path
                    d={d}
                    fill="none"
                    vectorEffect="non-scaling-stroke"
                    strokeWidth={2}
                    strokeLinecap="round"
                    className="stroke-primary"
                    initial={{ pathLength: 0.18, pathOffset: 0, opacity: 0 }}
                    animate={{ pathOffset: [0, 0.82], opacity: [0, 1, 0] }}
                    transition={{ duration: 1.2, repeat: Infinity, ease: 'easeInOut', delay: i * 0.18 }}
                  />
                )}
              </g>
            )
          })}
        </svg>
        </div>

        {/* Fusion */}
        <Node
          icon={phase === 'processing' ? <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.75} /> : phase === 'done' ? <Check className="h-4 w-4" strokeWidth={2.25} /> : <Layers className="h-4 w-4" strokeWidth={1.5} />}
          title="Fusion"
          detail={fusionLabel}
          tone={phase === 'processing' ? 'active' : phase === 'done' || allCaptured ? 'ready' : 'idle'}
        />

        <svg className="h-2 w-full" viewBox="0 0 100 10" preserveAspectRatio="none">
          <line x1="0" y1="5" x2="100" y2="5" vectorEffect="non-scaling-stroke" strokeWidth={1.25} className={phase === 'done' ? 'stroke-success/50' : 'stroke-border-strong'} />
        </svg>

        {/* Decision */}
        <Node icon={<ShieldCheck className="h-4 w-4" strokeWidth={1.5} />} title="Decision" detail={decisionLabel} tone={phase === 'done' ? 'ready' : 'idle'} />
      </div>
    </div>
  )
}

function Node({ icon, title, detail, tone }: { icon: ReactNode; title: string; detail: string; tone: 'idle' | 'active' | 'ready' }) {
  return (
    <motion.div
      layout
      className={cn(
        'flex flex-col items-center gap-1 rounded-xl border px-2.5 py-2 text-center transition-colors duration-300 sm:px-3',
        tone === 'active' ? 'border-primary/60 bg-primary/10 text-primary' : tone === 'ready' ? 'border-success/40 bg-success/10 text-success' : 'border-border-strong bg-raised/60 text-muted-foreground',
      )}
    >
      {icon}
      <span className="text-xs font-semibold text-foreground">{title}</span>
      <span className="text-[11px] leading-none">{detail}</span>
    </motion.div>
  )
}
