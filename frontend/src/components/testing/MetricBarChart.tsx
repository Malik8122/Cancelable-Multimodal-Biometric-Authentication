import { motion } from 'motion/react'
import type { Modality, ModalityMetricsResponse } from '../../api/types'

const COLORS: Record<Modality, string> = {
  face: 'var(--color-primary)',
  voice: 'var(--color-warning)',
  hand: 'var(--color-success)',
}

export function MetricBarChart({
  title,
  metricKey,
  metrics,
  percent = true,
}: {
  title: string
  metricKey: 'accuracy' | 'eer' | 'auc'
  metrics: Partial<Record<Modality, ModalityMetricsResponse>>
  percent?: boolean
}) {
  const modalities: Modality[] = ['face', 'voice', 'hand']
  const rows = modalities
    .map((m) => ({ modality: m, value: metrics[m]?.metrics[metricKey] }))
    .filter((r): r is { modality: Modality; value: number } => r.value !== undefined)

  if (rows.length === 0) {
    return (
      <div className="surface-card p-5 text-xs text-muted-foreground">
        {title}: no real data available yet.
      </div>
    )
  }

  const max = percent ? 1 : Math.max(...rows.map((r) => r.value), 1)

  return (
    <div className="surface-card p-5">
      <p className="mb-4 text-eyebrow text-muted-foreground">{title}</p>
      <div className="space-y-3">
        {rows.map((row) => (
          <div key={row.modality}>
            <div className="mb-1 flex items-center justify-between font-mono text-[10px] uppercase">
              <span className="text-muted-foreground">{row.modality}</span>
              <span className="text-foreground">{percent ? `${(row.value * 100).toFixed(2)}%` : row.value.toFixed(4)}</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-muted">
              <motion.div
                className="h-full rounded-full"
                style={{ backgroundColor: COLORS[row.modality] }}
                initial={{ width: 0 }}
                animate={{ width: `${(row.value / max) * 100}%` }}
                transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
