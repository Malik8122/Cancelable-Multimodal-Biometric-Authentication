import { motion } from 'motion/react'
import { useMemo } from 'react'

interface Props {
  seed: number
  brightness: number // 0..1, how many windows read as lit
  className?: string
}

// A restrained architectural illustration - glass-and-steel massing with a
// grid of window lights - standing in for photography we have no rights to
// use. Which windows read as "lit" is seeded per building so it's stable
// across re-renders, not Math.random() on every paint.
export function BuildingSilhouette({ seed, brightness, className }: Props) {
  const cols = 5
  const rows = 10
  const windows = useMemo(() => {
    let s = seed * 9301 + 49297
    const next = () => {
      s = (s * 9301 + 49297) % 233280
      return s / 233280
    }
    return Array.from({ length: cols * rows }, () => next())
  }, [seed])

  const w = 120
  const h = 260
  const pad = 14
  const cellW = (w - pad * 2) / cols
  const cellH = (h - pad * 2 - 18) / rows

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className={className} role="presentation" aria-hidden="true">
      {/* Parapet / roofline */}
      <rect x={pad - 4} y={4} width={w - (pad - 4) * 2} height={10} rx={1.5} fill="var(--color-border)" opacity={0.6} />
      {/* Facade */}
      <rect
        x={pad - 4}
        y={14}
        width={w - (pad - 4) * 2}
        height={h - 14}
        rx={2}
        fill="var(--color-card)"
        stroke="var(--color-border)"
        strokeWidth={1}
      />
      {windows.map((v, i) => {
        const col = i % cols
        const row = Math.floor(i / cols)
        const isLit = v < 0.35 + brightness * 0.5
        return (
          <motion.rect
            key={i}
            x={pad + col * cellW + cellW * 0.14}
            y={pad + 10 + row * cellH + cellH * 0.18}
            width={cellW * 0.72}
            height={cellH * 0.64}
            rx={0.6}
            fill={isLit ? 'var(--color-primary)' : 'var(--color-border)'}
            animate={{ opacity: isLit ? 0.35 + brightness * 0.5 : 0.35 }}
            transition={{ duration: 0.6 }}
          />
        )
      })}
    </svg>
  )
}
