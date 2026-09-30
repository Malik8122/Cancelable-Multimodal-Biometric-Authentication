import { AlertCircle, Check } from 'lucide-react'
import type { ReactNode } from 'react'
import type { RecordingQuality } from '../../api/types'

// The wording of the two non-green cases (shared with the backend's messages).
export const FAIR_MESSAGE = 'Your recordings are usable, but quality is lower than recommended.'
export const POOR_MESSAGE =
  'Each sentence passed the audio check, but the two did not sound consistent enough to build one voice template. Record both again in one sitting - same place, same distance from the microphone, your normal speaking voice.'

const STYLE: Record<RecordingQuality, { label: string; box: string; badge: string; Icon: typeof Check }> = {
  EXCELLENT: { label: 'Excellent', box: 'border-success/30 bg-success/10', badge: 'text-success', Icon: Check },
  GOOD: { label: 'Good', box: 'border-success/30 bg-success/10', badge: 'text-success', Icon: Check },
  FAIR: { label: 'Fair', box: 'border-warning/30 bg-warning/10', badge: 'text-warning', Icon: AlertCircle },
  POOR: { label: 'Poor', box: 'border-danger/30 bg-danger/10', badge: 'text-danger', Icon: AlertCircle },
}

// "Recording Quality: Excellent | Good (green), Fair (yellow), Poor (red)" plus a message and the actions the band allows.
// It is the ONLY feedback for the voice consistency check - there is no separate "retry required" screen.
export function RecordingQualityPanel({
  quality,
  message,
  children,
}: {
  quality: RecordingQuality
  message?: string
  children?: ReactNode
}) {
  const { label, box, badge, Icon } = STYLE[quality]
  return (
    <div className={`mb-4 rounded-xl border p-4 ${box}`} role="status">
      <p className="mb-1 flex items-center gap-2 text-sm">
        <span className="text-muted-foreground">Recording Quality</span>
        <span className={`flex items-center gap-1 font-medium ${badge}`}>
          <Icon className="h-3.5 w-3.5" strokeWidth={2} /> {label}
        </span>
      </p>
      {message && <p className={`text-xs ${quality === 'POOR' || quality === 'FAIR' ? badge : 'text-foreground/80'}`}>{message}</p>}
      {children && <div className="mt-3 flex flex-wrap gap-2">{children}</div>}
    </div>
  )
}
