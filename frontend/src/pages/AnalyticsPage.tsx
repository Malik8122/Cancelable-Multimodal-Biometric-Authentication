import { motion } from 'motion/react'
import { Activity, AlertTriangle, Building2, CheckCircle2, Clock, Hand, Mic, ScanFace, Users } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { getSystemAuditHistory } from '../api/client'
import type { AuditLogEntry, Modality } from '../api/types'
import { useBuildings } from '../context/BuildingsContext'
import { useCountUp } from '../hooks/useCountUp'
import { PageHeader } from '../components/layout/PageHeader'

const MODALITY_ICON: Record<string, typeof ScanFace> = { face: ScanFace, voice: Mic, hand: Hand }

function StatWidget({ icon: Icon, label, value, suffix = '' }: { icon: typeof Activity; label: string; value: number; suffix?: string }) {
  const animated = useCountUp(value, 1)
  return (
    <div className="surface-card relative p-5">
      <div className="relative flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-primary/30 bg-primary/10">
          <Icon className="h-4 w-4 text-primary" />
        </div>
        <div>
          <p className="text-eyebrow text-muted-foreground">{label}</p>
          <p className="font-mono text-xl font-bold text-foreground">
            {animated.toFixed(suffix === '%' || suffix === 'ms' ? 0 : 0)}
            {suffix}
          </p>
        </div>
      </div>
    </div>
  )
}

export function AnalyticsPage() {
  const { getBuilding } = useBuildings()
  const [entries, setEntries] = useState<AuditLogEntry[] | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    getSystemAuditHistory(200).then((result) => {
      if (result) setEntries(result.entries)
      else setError(true)
    })
  }, [])

  const stats = useMemo(() => {
    if (!entries) return null
    const todayStart = new Date()
    todayStart.setHours(0, 0, 0, 0)
    const today = entries.filter((e) => new Date(e.timestamp) >= todayStart)
    // ENROLLMENT_REQUIRED is its own state, never a failed authentication: nothing biometric was evaluated.
    const evaluated = entries.filter((e) => e.authentication_state !== 'ENROLLMENT_REQUIRED')
    const successCount = evaluated.filter((e) => e.authentication_state === 'ACCESS_GRANTED').length
    const failedCount = evaluated.length - successCount
    const enrollmentRequiredCount = entries.length - evaluated.length
    const avgLatency = entries.length ? Math.round(entries.reduce((sum, e) => sum + e.latency_ms, 0) / entries.length) : 0

    const modalityCounts: Partial<Record<Modality, number>> = {}
    for (const entry of entries) {
      for (const modality of entry.modality_list) {
        modalityCounts[modality] = (modalityCounts[modality] ?? 0) + 1
      }
    }
    const mostUsed: Modality | '-' = (Object.entries(modalityCounts) as [Modality, number][]).sort((a, b) => b[1] - a[1])[0]?.[0] ?? '-'

    const buildingCounts: Record<string, number> = {}
    for (const entry of entries) {
      if (entry.building_id) buildingCounts[entry.building_id] = (buildingCounts[entry.building_id] ?? 0) + 1
    }
    const recentBuildings = Object.entries(buildingCounts).sort((a, b) => b[1] - a[1])

    const recentUsers = [...new Set(entries.map((e) => e.user_id))].slice(0, 8)

    return {
      todayCount: today.length,
      successRate: evaluated.length ? Math.round((successCount / evaluated.length) * 100) : 0,
      failedCount,
      enrollmentRequiredCount,
      avgLatency,
      mostUsed,
      recentBuildings,
      recentUsers,
    }
  }, [entries])

  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6 sm:py-12">
      <PageHeader
        eyebrow="Security analytics"
        title="Operational Intelligence"
        description="Live figures from the backend audit log: attempts, decisions, factors and latency. Nothing here is simulated."
      />

      {error && (
        <div className="mb-6 flex items-center gap-2 rounded-lg border border-warning/40 bg-warning/10 p-4 text-sm text-warning">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          Backend unreachable - live audit data cannot be loaded.
        </div>
      )}

      {!error && !entries && <div className="surface-card h-40 animate-pulse" aria-label="Loading audit data" />}

      {stats && (
        <>
          <div className="mb-8 grid grid-cols-2 gap-4 sm:grid-cols-5">
            <StatWidget icon={Clock} label="Today's Attempts" value={stats.todayCount} />
            <StatWidget icon={CheckCircle2} label="Success Rate" value={stats.successRate} suffix="%" />
            <StatWidget icon={AlertTriangle} label="Denied Attempts" value={stats.failedCount} />
            <StatWidget icon={AlertTriangle} label="Enrollment Required" value={stats.enrollmentRequiredCount} />
            <StatWidget icon={Activity} label="Avg Latency" value={stats.avgLatency} suffix="ms" />
          </div>

          <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="surface-card relative p-5">
              <p className="relative mb-3 text-eyebrow text-muted-foreground">Most-Used Modality</p>
              <div className="relative flex items-center gap-3">
                {(() => {
                  const Icon = MODALITY_ICON[stats.mostUsed]
                  return Icon ? <Icon className="h-8 w-8 text-primary" /> : null
                })()}
                <span className="font-mono text-lg font-semibold text-foreground uppercase">{stats.mostUsed}</span>
              </div>
            </div>

            <div className="surface-card relative p-5">
              <p className="relative mb-3 flex items-center gap-1.5 text-eyebrow text-muted-foreground">
                <Building2 className="h-3 w-3" /> Recent Facilities
              </p>
              <ul className="relative space-y-1.5">
                {stats.recentBuildings.length === 0 && <li className="text-xs text-muted-foreground">No facility-tagged attempts yet.</li>}
                {stats.recentBuildings.map(([id, count]) => (
                  <li key={id} className="flex items-center justify-between font-mono text-xs">
                    <span className="text-foreground">{getBuilding(id)?.name ?? id}</span>
                    <span className="text-muted-foreground">{count}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="mb-8 surface-card relative p-5">
            <p className="relative mb-3 flex items-center gap-1.5 text-eyebrow text-muted-foreground">
              <Users className="h-3 w-3" /> Recent Operators
            </p>
            <div className="relative flex flex-wrap gap-2">
              {stats.recentUsers.map((user) => (
                <span key={user} className="rounded-full border border-border bg-muted px-3 py-1 font-mono text-[10px] text-muted-foreground">
                  {user}
                </span>
              ))}
            </div>
          </div>

          <div>
            <p className="mb-3 text-eyebrow text-muted-foreground">Audit Activity</p>
            <div className="surface-card overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-border text-muted-foreground">
                    <th className="px-4 py-2 font-mono font-normal uppercase">Time</th>
                    <th className="px-4 py-2 font-mono font-normal uppercase">User</th>
                    <th className="px-4 py-2 font-mono font-normal uppercase">Facility</th>
                    <th className="px-4 py-2 font-mono font-normal uppercase">Factors</th>
                    <th className="px-4 py-2 font-mono font-normal uppercase">Decision</th>
                    <th className="px-4 py-2 font-mono font-normal uppercase">Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {entries!.slice(0, 20).map((entry, i) => (
                    <motion.tr
                      key={entry.audit_id}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: i * 0.02 }}
                      className="border-b border-border/50 text-muted-foreground"
                    >
                      <td className="px-4 py-2 font-mono">{new Date(entry.timestamp).toLocaleTimeString()}</td>
                      <td className="px-4 py-2 font-mono">{entry.user_id}</td>
                      <td className="px-4 py-2">{entry.building_id ? (getBuilding(entry.building_id)?.name ?? entry.building_id) : '-'}</td>
                      <td className="px-4 py-2">{entry.modality_list.join(' + ')}</td>
                      <td
                        className={`px-4 py-2 font-mono ${
                          entry.authentication_state === 'ACCESS_GRANTED'
                            ? 'text-success'
                            : entry.authentication_state === 'ENROLLMENT_REQUIRED'
                              ? 'text-warning'
                              : 'text-danger'
                        }`}
                      >
                        {entry.authentication_state === 'ACCESS_GRANTED'
                          ? 'GRANTED'
                          : entry.authentication_state === 'ENROLLMENT_REQUIRED'
                            ? 'ENROLLMENT REQUIRED'
                            : 'DENIED'}
                      </td>
                      <td className="px-4 py-2 font-mono">{entry.latency_ms}ms</td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
              {entries!.length === 0 && <p className="p-4 text-center text-xs text-muted-foreground">No authentication attempts recorded yet.</p>}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
