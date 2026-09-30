import { AlertCircle, Check } from 'lucide-react'
import type { EnrollmentStatus } from '../../api/types'

export const STATUS_LABEL: Record<EnrollmentStatus, string> = {
  NOT_REGISTERED: 'Not registered',
  REGISTERED: 'Registered',
  UPDATED: 'Updated',
  RETRY_REQUIRED: 'Re-record needed',
}

// One pill for the four per-modality enrollment states. RETRY_REQUIRED only ever appears for voice: the two
// recordings were not consistent and nothing was stored.
export function EnrollmentStatusPill({ status }: { status: EnrollmentStatus | undefined }) {
  const value = status ?? 'NOT_REGISTERED'
  if (value === 'REGISTERED' || value === 'UPDATED') {
    return (
      <span className="flex items-center gap-1 rounded-full border border-success/30 bg-success/10 px-2.5 py-0.5 text-[11px] font-medium text-success">
        <Check className="h-3 w-3" strokeWidth={2} /> {STATUS_LABEL[value]}
      </span>
    )
  }
  if (value === 'RETRY_REQUIRED') {
    return (
      <span className="flex items-center gap-1 rounded-full border border-warning/30 bg-warning/10 px-2.5 py-0.5 text-[11px] font-medium text-warning">
        <AlertCircle className="h-3 w-3" strokeWidth={2} /> {STATUS_LABEL[value]}
      </span>
    )
  }
  return (
    <span className="rounded-full border border-warning/30 bg-warning/10 px-2.5 py-0.5 text-[11px] font-medium text-warning">
      {STATUS_LABEL[value]}
    </span>
  )
}
