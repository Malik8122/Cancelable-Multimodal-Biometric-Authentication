import { motion, usePresenceData, type Variants } from 'motion/react'
import type { ReactNode } from 'react'
import { EASE_IN, EASE_OUT } from '../../lib/motion'

/**
 * Direction of a navigation along the authentication flow (Campus -> Checkpoint -> Register/Authenticate -> Result):
 * 1 = deeper, -1 = back, 0 = anything else (sections like Testing or Analytics). App.tsx computes it and passes it to
 * AnimatePresence as `custom`, so both the entering and the leaving page read the same value.
 */
export type NavDirection = -1 | 0 | 1

const variants: Variants = {
  enter: (d: NavDirection) => ({ opacity: 0, x: d * 48, y: d === 0 ? 8 : 0 }),
  center: { opacity: 1, x: 0, y: 0, transition: { duration: 0.34, ease: EASE_OUT } },
  exit: (d: NavDirection) => ({ opacity: 0, x: d * -48, y: d === 0 ? -4 : 0, transition: { duration: 0.18, ease: EASE_IN } }),
}

// Deeper into the flow slides forward, back slides backward, other pages cross-fade. No blur (expensive next to the
// camera and MediaPipe). Reduced motion keeps only the fade (MotionConfig reducedMotion="user").
export function PageTransition({ children }: { children: ReactNode }) {
  const direction = (usePresenceData() as NavDirection | undefined) ?? 0
  return (
    <motion.div custom={direction} variants={variants} initial="enter" animate="center" exit="exit">
      {children}
    </motion.div>
  )
}
