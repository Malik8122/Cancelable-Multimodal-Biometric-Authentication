import type { Modality } from '../../api/types'
import { MODALITY_ICON } from '../../config/modalityIcons'
import { cn } from '../../lib/utils'

type Tone = 'default' | 'active' | 'success' | 'warning' | 'danger' | 'muted'

const TONE: Record<Tone, string> = {
  default: 'border-primary/25 bg-primary/10 text-primary',
  active: 'border-primary/50 bg-primary/15 text-primary',
  success: 'border-success/35 bg-success/10 text-success',
  warning: 'border-warning/35 bg-warning/10 text-warning',
  danger: 'border-danger/35 bg-danger/10 text-danger',
  muted: 'border-border bg-raised/60 text-muted-foreground',
}

const SIZE = { sm: 'h-8 w-8 rounded-lg [&>svg]:h-4 [&>svg]:w-4', md: 'h-10 w-10 rounded-xl [&>svg]:h-5 [&>svg]:w-5', lg: 'h-12 w-12 rounded-xl [&>svg]:h-6 [&>svg]:w-6' }

/** The modality's icon tile. Decorative: the modality is always also named in text next to it. */
export function ModalityBadge({ modality, tone = 'default', size = 'md', className }: { modality: Modality; tone?: Tone; size?: keyof typeof SIZE; className?: string }) {
  const Icon = MODALITY_ICON[modality]
  return (
    <span aria-hidden className={cn('flex shrink-0 items-center justify-center border transition-colors', SIZE[size], TONE[tone], className)}>
      <Icon strokeWidth={1.5} />
    </span>
  )
}
