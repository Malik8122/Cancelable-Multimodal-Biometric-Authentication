import { animate } from 'framer-motion'
import { useEffect, useState } from 'react'
import { useReducedMotion } from './useReducedMotion'

/** Animates a displayed number from 0 to `target` - purely cosmetic, the
 * real value (`target`) is always what gets used everywhere else. */
export function useCountUp(target: number, duration = 0.8): number {
  const reducedMotion = useReducedMotion()
  const [value, setValue] = useState(reducedMotion ? target : 0)

  useEffect(() => {
    if (reducedMotion) {
      setValue(target)
      return
    }
    const controls = animate(0, target, {
      duration,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: setValue,
    })
    return () => controls.stop()
  }, [target, duration, reducedMotion])

  return value
}
