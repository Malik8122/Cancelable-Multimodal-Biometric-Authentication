import { motion } from 'motion/react'
import type { Building } from '../../config/buildings'

const STYLE: Record<Building['clearanceLevel'], string> = {
  III: 'text-primary border-primary/30 bg-primary/10',
  IV: 'text-warning border-warning/30 bg-warning/10',
  V: 'text-danger border-danger/30 bg-danger/10',
}

export function ClearanceBadge({ level }: { level: Building['clearanceLevel'] }) {
  return (
    <motion.span
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium tracking-wide ${STYLE[level]}`}
    >
      Clearance Level {level}
    </motion.span>
  )
}
