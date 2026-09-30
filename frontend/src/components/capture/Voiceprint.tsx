import { motion } from 'motion/react'
import { cn } from '../../lib/utils'

/**
 * A voiceprint strip: the loudness envelope of an ACCEPTED recording (lib/biometricVisuals.ts::voiceprintLevels,
 * decoded from the actual WAV). The bars grow in left to right, like the recording being "printed". It is a picture of
 * the recording's loudness over time, not the speaker embedding.
 */
export function Voiceprint({ levels, compact, className, label }: { levels: number[]; compact?: boolean; className?: string; label: string }) {
  return (
    <motion.div
      role="img"
      aria-label={label}
      className={cn('flex items-center justify-center gap-[2px]', compact ? 'h-6' : 'h-16 rounded-xl border border-success/35 bg-success/[0.06] px-3', className)}
      initial="hidden"
      animate="show"
      variants={{ hidden: {}, show: { transition: { staggerChildren: compact ? 0.004 : 0.012 } } }}
    >
      {levels.map((level, i) => (
        <motion.span
          key={i}
          className={cn('w-[3px] origin-center rounded-full bg-success', compact ? 'h-6' : 'h-12')}
          variants={{ hidden: { scaleY: 0.05, opacity: 0.3 }, show: { scaleY: Math.max(0.08, level), opacity: 0.55 + 0.45 * level } }}
          transition={{ type: 'spring', stiffness: 320, damping: 24 }}
        />
      ))}
    </motion.div>
  )
}
