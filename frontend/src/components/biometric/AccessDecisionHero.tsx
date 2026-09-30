import { motion } from 'motion/react'
import { ShieldCheck, ShieldX } from 'lucide-react'
import { EASE_OUT } from '../../lib/motion'

/**
 * The final decision, stated once and clearly: a ring that closes around the verdict icon, then the verdict in words.
 * Granted and denied differ in icon, wording AND color - never color alone. No flashes, doors or confetti.
 */
export function AccessDecisionHero({ authenticated, subtitle }: { authenticated: boolean; subtitle?: string }) {
  const Icon = authenticated ? ShieldCheck : ShieldX
  const tone = authenticated ? 'text-success' : 'text-danger'
  const stroke = authenticated ? 'stroke-success' : 'stroke-danger'

  return (
    <div className="flex flex-col items-center py-8 text-center sm:py-10" role="status" aria-live="polite">
      <div className="relative mb-6 h-28 w-28">
        <svg aria-hidden viewBox="0 0 100 100" className="absolute inset-0 h-full w-full -rotate-90">
          <circle cx="50" cy="50" r="46" fill="none" strokeWidth="2" className="stroke-border-strong" />
          <motion.circle
            cx="50"
            cy="50"
            r="46"
            fill="none"
            strokeWidth="2.5"
            strokeLinecap="round"
            className={stroke}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.7, ease: EASE_OUT }}
          />
        </svg>
        <motion.div
          initial={{ opacity: 0, scale: 0.7 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: 'spring', stiffness: 260, damping: 22, delay: 0.35 }}
          className={`absolute inset-3 flex items-center justify-center rounded-full ${authenticated ? 'bg-success/10' : 'bg-danger/10'}`}
        >
          <Icon className={`h-11 w-11 ${tone}`} strokeWidth={1.5} aria-hidden />
        </motion.div>
      </div>

      <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.45 }} className={`text-eyebrow mb-1 ${tone}`}>
        {authenticated ? 'Access granted' : 'Access denied'}
      </motion.p>
      <motion.h1
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.35, ease: EASE_OUT }}
        className="text-page-title text-foreground"
      >
        {authenticated ? 'Identity Verified' : 'Identity Not Verified'}
      </motion.h1>
      {subtitle && (
        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }} className="mt-2 max-w-md text-sm text-muted-foreground">
          {subtitle}
        </motion.p>
      )}
    </div>
  )
}
