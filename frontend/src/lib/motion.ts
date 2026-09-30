import type { Transition, Variants } from 'motion/react'

// One motion vocabulary for the whole console. Motion explains state (captured, verifying, verified, failed) - it is
// never decoration. Durations: 150-250 ms for micro-interactions, up to ~400 ms for page-level changes. Users who ask
// for reduced motion get instant state changes (MotionConfig reducedMotion="user" in App.tsx).

/** Ease-out for things entering, ease-in for things leaving. */
export const EASE_OUT = [0.22, 1, 0.36, 1] as const
export const EASE_IN = [0.4, 0, 1, 1] as const

export const DURATION = { fast: 0.15, base: 0.22, slow: 0.36 } as const

export const quick: Transition = { duration: DURATION.base, ease: EASE_OUT }
export const spring: Transition = { type: 'spring', stiffness: 380, damping: 32, mass: 0.8 }

/** Fade + small rise, for a block that appears (a status message, a result section). */
export const rise: Variants = {
  hidden: { opacity: 0, y: 8 },
  show: { opacity: 1, y: 0, transition: { duration: DURATION.slow, ease: EASE_OUT } },
  exit: { opacity: 0, y: -4, transition: { duration: DURATION.fast, ease: EASE_IN } },
}

/** Parent that reveals its children one after another (60 ms apart). */
export const stagger: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06, delayChildren: 0.04 } },
}

/** Subtle press feedback for buttons and selectable cards (no layout shift - transform only). */
export const press = { whileHover: { y: -1 }, whileTap: { scale: 0.98 } } as const
