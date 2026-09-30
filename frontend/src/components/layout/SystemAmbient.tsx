import { motion } from 'motion/react'
import type { AmbientState } from '../../context/ambientState'
import { useAmbientState } from '../../hooks/useAmbient'

// The glow color per state. Low alpha on purpose: the backdrop should feel alive, never compete with content.
const GLOW: Record<AmbientState, string> = {
  idle: 'rgba(106, 147, 240, 0.14)',
  capturing: 'rgba(76, 195, 224, 0.13)',
  processing: 'rgba(150, 120, 240, 0.15)',
  success: 'rgba(63, 180, 124, 0.13)',
  failure: 'rgba(239, 107, 112, 0.12)',
}

/**
 * The console's backdrop: navy base, a soft top light that takes the color of the current system state, and a faint
 * technical grid. It only changes when the state changes (a slow 1.6 s cross-fade) - there is no looping animation.
 */
export function SystemAmbient() {
  const state = useAmbientState()
  return (
    <div className="app-backdrop" aria-hidden>
      <motion.div
        className="absolute inset-0"
        initial={false}
        animate={{ '--ambient-glow': GLOW[state] } as Record<string, string>}
        transition={{ duration: 1.6, ease: 'easeInOut' }}
        style={{
          background:
            'radial-gradient(1100px 560px at 50% -10%, var(--ambient-glow), transparent 68%), radial-gradient(700px 420px at 100% 100%, color-mix(in srgb, var(--color-info) 4%, transparent), transparent 70%)',
        }}
      />
    </div>
  )
}
