import type { Modality, ModalityMetricsResponse } from '../../api/types'
import { useCountUp } from '../../hooks/useCountUp'

function Counter({ value, suffix = '%' }: { value: number; suffix?: string }) {
  const animated = useCountUp(value * 100, 1.2)
  return (
    <span className="font-mono text-3xl font-bold text-foreground">
      {animated.toFixed(2)}
      {suffix}
    </span>
  )
}

export function MetricCounterCard({ modality, metrics }: { modality: Modality; metrics: ModalityMetricsResponse }) {
  const accuracy = metrics.metrics.accuracy
  const eer = metrics.metrics.eer
  const auc = metrics.metrics.auc

  return (
    <div className="surface-card relative p-5">
      <p className="relative mb-3 font-mono text-xs tracking-[0.15em] text-primary uppercase">{modality}</p>
      {!metrics.available ? (
        <p className="relative text-sm text-muted-foreground">Not evaluated yet.</p>
      ) : (
        <div className="relative">
          <div className="mb-3">
            <p className="text-eyebrow text-muted-foreground">Accuracy</p>
            {accuracy !== undefined ? <Counter value={accuracy} /> : <span className="text-sm text-muted-foreground">-</span>}
          </div>
          <div className="grid grid-cols-2 gap-3 border-t border-border pt-3 font-mono text-[11px]">
            <div>
              <p className="text-muted-foreground uppercase">EER</p>
              <p className="text-foreground">{eer !== undefined ? `${(eer * 100).toFixed(2)}%` : 'Not evaluated yet'}</p>
            </div>
            <div>
              <p className="text-muted-foreground uppercase">ROC-AUC</p>
              <p className="text-foreground">{auc !== undefined ? auc.toFixed(4) : 'Not evaluated yet'}</p>
            </div>
            <div>
              <p className="text-muted-foreground uppercase">FAR</p>
              <p className="text-foreground">
                {metrics.metrics.far !== undefined ? `${(metrics.metrics.far * 100).toFixed(2)}%` : 'Not evaluated yet'}
              </p>
            </div>
            <div>
              <p className="text-muted-foreground uppercase">FRR</p>
              <p className="text-foreground">
                {metrics.metrics.frr !== undefined ? `${(metrics.metrics.frr * 100).toFixed(2)}%` : 'Not evaluated yet'}
              </p>
            </div>
          </div>
          {metrics.calibrated && (
            <div className="mt-3 rounded-md border border-primary/30 bg-primary/5 px-2 py-1.5 font-mono text-[9px] text-primary uppercase">
              Calibrated threshold: {metrics.metrics.calibrated_threshold?.toFixed(3)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
