import { motion } from 'motion/react'
import { Check, Hand, Layers, Mic, ScanFace } from 'lucide-react'

type SetModality = 'face' | 'voice' | 'hand'

const META: Record<SetModality, { label: string; icon: typeof ScanFace }> = {
  face: { label: 'Face', icon: ScanFace },
  voice: { label: 'Voice', icon: Mic },
  hand: { label: 'Hand', icon: Hand },
}

/** Ticks of the registration animation: capture + one per template set + pool ready. */
export function animationTicks(poolSize: number) {
  return poolSize + 2
}

/**
 * Registration animation:
 *   Capture Face / Voice / Hand -> Generate Template Set 1..N
 *   (each set = one template per modality) -> Template Set Pool -> Set 1 Active, others Standby
 */
export function TemplateSetAnimation({
  modalities,
  poolSize,
  tick,
}: {
  modalities: SetModality[]
  poolSize: number
  tick: number
}) {
  const captured = tick >= 1
  const poolReady = tick >= animationTicks(poolSize)

  return (
    <div className="mx-auto max-w-2xl space-y-4">
      <div className="flex flex-wrap justify-center gap-2">
        {modalities.map((m) => {
          const { icon: Icon, label } = META[m]
          return (
            <span
              key={m}
              className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
                captured ? 'border-success/25 bg-success/5 text-success' : 'border-border text-muted-foreground'
              }`}
            >
              <Icon className="h-3.5 w-3.5" strokeWidth={1.5} />
              {captured ? `${label} captured` : `Capture ${label}`}
              {captured && <Check className="h-3 w-3" strokeWidth={2} />}
            </span>
          )
        })}
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {Array.from({ length: poolSize }, (_, i) => {
          const version = i + 1
          const generated = tick >= 1 + version
          const generating = tick === version
          const isActive = poolReady && version === 1
          return (
            <motion.div
              key={version}
              animate={generating ? { scale: [1, 1.03, 1] } : { scale: 1 }}
              className={`rounded-xl border p-3.5 transition-colors ${
                isActive
                  ? 'border-success/40 bg-success/10'
                  : poolReady
                    ? 'border-primary/25 bg-primary/5'
                    : generated
                      ? 'border-primary/30 bg-primary/5'
                      : 'border-border bg-card/30'
              }`}
            >
              <div className="mb-2 flex items-center justify-between">
                <span className={`text-sm font-medium ${generated ? 'text-foreground' : 'text-muted-foreground'}`}>
                  {generated ? `Template Set ${version}` : `Generate Template Set ${version}`}
                </span>
                {poolReady && (
                  <span className={`text-[11px] font-medium ${isActive ? 'text-success' : 'text-primary'}`}>
                    {isActive ? 'Active' : 'Standby'}
                  </span>
                )}
              </div>
              <ul className="space-y-1">
                {modalities.map((m) => {
                  const { icon: Icon, label } = META[m]
                  return (
                    <li
                      key={m}
                      className={`flex items-center gap-2 text-xs ${generated ? 'text-foreground' : 'text-muted-foreground/60'}`}
                    >
                      <Icon className="h-3.5 w-3.5 text-primary" strokeWidth={1.5} />
                      {label} Template V{version}
                      {generated && <Check className="ml-auto h-3 w-3 text-success" strokeWidth={2} />}
                    </li>
                  )
                })}
              </ul>
            </motion.div>
          )
        })}
      </div>

      <p className={`flex items-center justify-center gap-1.5 text-xs ${poolReady ? 'text-success' : 'text-muted-foreground'}`}>
        <Layers className="h-3.5 w-3.5" strokeWidth={1.5} />
        {poolReady ? `Template set pool ready - Set 1 active, Sets 2-${poolSize} standby` : 'Template set pool'}
      </p>
    </div>
  )
}
