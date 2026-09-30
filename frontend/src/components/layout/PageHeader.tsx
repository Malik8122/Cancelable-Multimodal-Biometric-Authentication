import { motion } from 'motion/react'
import type { ReactNode } from 'react'
import { rise, stagger } from '../../lib/motion'

/** The one page-header pattern of the console: eyebrow, title, optional description and actions. */
export function PageHeader({ eyebrow, title, description, actions }: { eyebrow: string; title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <motion.header variants={stagger} initial="hidden" animate="show" className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-2xl">
        <motion.p variants={rise} className="text-eyebrow mb-1.5 text-info">
          {eyebrow}
        </motion.p>
        <motion.h1 variants={rise} className="text-page-title text-foreground">
          {title}
        </motion.h1>
        {description && (
          <motion.p variants={rise} className="mt-2 text-sm leading-relaxed text-muted-foreground">
            {description}
          </motion.p>
        )}
      </div>
      {actions && <motion.div variants={rise}>{actions}</motion.div>}
    </motion.header>
  )
}
