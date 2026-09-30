import { motion, useMotionValue, useTransform, animate } from 'motion/react'
import { useEffect, useState } from 'react'
import { useReducedMotion } from '../../hooks/useReducedMotion'

interface Props {
  value: number // 0..1
  label: string
  colorVar?: string // css color, defaults to primary
  size?: number
}

export function RadialGauge({ value, label, colorVar = 'var(--color-primary)', size = 140 }: Props) {
  const reducedMotion = useReducedMotion()
  const radius = size / 2 - 10
  const circumference = 2 * Math.PI * radius
  const motionValue = useMotionValue(0)
  const [displayPercent, setDisplayPercent] = useState(0)
  const dashoffset = useTransform(motionValue, (v) => circumference * (1 - v))

  useEffect(() => {
    const controls = animate(motionValue, value, {
      duration: reducedMotion ? 0 : 1.2,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (v) => setDisplayPercent(Math.round(v * 100)),
    })
    return () => controls.stop()
  }, [value, motionValue, reducedMotion])

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="var(--color-border)" strokeWidth={8} />
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={colorVar}
            strokeWidth={8}
            strokeLinecap="round"
            strokeDasharray={circumference}
            style={{ strokeDashoffset: dashoffset, filter: `drop-shadow(0 0 6px ${colorVar})` }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="font-mono text-2xl font-semibold text-foreground">{displayPercent}%</span>
        </div>
      </div>
      <span className="font-mono text-[10px] tracking-wider text-muted-foreground uppercase">{label}</span>
    </div>
  )
}
