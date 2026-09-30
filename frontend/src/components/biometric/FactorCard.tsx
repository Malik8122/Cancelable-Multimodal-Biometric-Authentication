import { motion } from 'motion/react'
import { Check, ChevronRight } from 'lucide-react'
import type { ReactNode } from 'react'
import type { Modality } from '../../api/types'
import { MODALITY_LABEL } from '../../config/buildings'
import { cn } from '../../lib/utils'
import { ModalityBadge } from './ModalityBadge'

/**
 * One presented factor in the capture deck (Authenticate page). It shares its layoutId with the expanded capture
 * panel, so opening it visibly grows the card into the capture experience and closing it shrinks back into the card.
 * Once captured, the card shows what was captured (face frame, voiceprints, gesture trajectory).
 */
export function FactorCard({
  modality,
  captured,
  summary,
  onOpen,
}: {
  modality: Modality
  captured: boolean
  summary?: ReactNode
  onOpen: () => void
}) {
  return (
    <motion.button
      type="button"
      layoutId={`capture-${modality}`}
      onClick={onOpen}
      style={{ borderRadius: 24 }}
      whileHover={{ y: -2 }}
      whileTap={{ scale: 0.985 }}
      aria-label={`${MODALITY_LABEL[modality]}: ${captured ? 'captured - open to review or retake' : 'not captured - open to capture'}`}
      className={cn(
        'surface-card group flex w-full cursor-pointer flex-col gap-3 p-4 text-left transition-colors duration-300',
        captured ? 'border-success/40' : 'hover:border-primary/50',
      )}
    >
      <motion.div layout="position" className="flex w-full items-center justify-between gap-2">
        <span className="flex items-center gap-3">
          <ModalityBadge modality={modality} tone={captured ? 'success' : 'default'} />
          <span>
            <span className="text-card-title block text-foreground">{MODALITY_LABEL[modality]}</span>
            <span className={cn('text-meta flex items-center gap-1', captured ? 'text-success' : 'text-muted-foreground')}>
              {captured ? (
                <>
                  <Check className="h-3.5 w-3.5" strokeWidth={2.25} aria-hidden /> Captured
                </>
              ) : (
                'Not captured yet'
              )}
            </span>
          </span>
        </span>
        <ChevronRight className="h-4 w-4 text-muted-foreground transition-transform duration-200 group-hover:translate-x-0.5" strokeWidth={1.75} aria-hidden />
      </motion.div>
      <motion.div layout="position" className="flex h-16 w-full items-center justify-center rounded-xl border border-border bg-black/30">
        {captured && summary ? summary : <span className="text-xs text-muted-foreground">Open to capture</span>}
      </motion.div>
    </motion.button>
  )
}
