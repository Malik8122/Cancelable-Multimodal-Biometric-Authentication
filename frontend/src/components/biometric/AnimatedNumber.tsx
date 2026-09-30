import { motion, useSpring, useTransform } from 'motion/react'
import { useEffect } from 'react'

/**
 * A REAL value (e.g. the measured verification time) that settles into place with a spring. The final number is
 * always exactly `value`; only the approach is animated. Reduced motion shows it immediately (MotionConfig).
 */
export function AnimatedNumber({ value, decimals = 0, suffix = '', className }: { value: number; decimals?: number; suffix?: string; className?: string }) {
  const spring = useSpring(0, { stiffness: 90, damping: 22 })
  const text = useTransform(spring, (v) => `${v.toFixed(decimals)}${suffix}`)
  useEffect(() => {
    spring.set(value)
  }, [spring, value])
  return (
    <>
      <motion.span aria-hidden className={className}>
        {text}
      </motion.span>
      <span className="sr-only">
        {value.toFixed(decimals)}
        {suffix}
      </span>
    </>
  )
}
