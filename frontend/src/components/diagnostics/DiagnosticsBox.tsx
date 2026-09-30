import { FlaskConical } from 'lucide-react'
import type { ReactNode } from 'react'
import { cn } from '../../lib/utils'

export type DiagnosticStatus = 'pass' | 'fail' | 'info'

export interface DiagnosticRow {
  label: string
  value: ReactNode
  status?: DiagnosticStatus
}

const MARK: Record<DiagnosticStatus, string> = { pass: 'PASS', fail: 'FAIL', info: '' }

/**
 * A compact developer-diagnostics block (shown only when the backend runs with DEBUG_SCORES - see
 * useDiagnosticsEnabled). Values are the backend's own measurements; nothing here is computed for display.
 */
export function DiagnosticsBox({ title, rows, footer, className }: { title: string; rows: DiagnosticRow[]; footer?: ReactNode; className?: string }) {
  return (
    <div className={cn('rounded-xl border border-processing/35 bg-processing/[0.05] p-3 text-xs', className)}>
      <p className="mb-2 flex items-center gap-1.5 font-medium text-processing">
        <FlaskConical className="h-3.5 w-3.5" strokeWidth={1.75} aria-hidden />
        {title}
        <span className="ml-auto rounded-full border border-border-strong px-1.5 py-px text-[10px] font-normal text-muted-foreground">dev only</span>
      </p>
      <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1">
        {rows.map((row) => (
          <div key={row.label} className="contents">
            <dt className="truncate text-muted-foreground">{row.label}</dt>
            <dd className="text-right font-mono tabular-nums text-foreground">
              {row.value}
              {row.status && row.status !== 'info' && (
                <span className={cn('ml-2 font-sans font-semibold', row.status === 'pass' ? 'text-success' : 'text-danger')}>{MARK[row.status]}</span>
              )}
            </dd>
          </div>
        ))}
      </dl>
      {footer && <div className="mt-2 border-t border-processing/25 pt-2 text-muted-foreground">{footer}</div>}
    </div>
  )
}
