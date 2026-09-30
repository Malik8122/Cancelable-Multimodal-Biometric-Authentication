import { ShieldAlert } from 'lucide-react'
import { useSession } from '../../context/SessionContext'

// Never fabricates a "connected" state, and never offers a fake-data fallback - if the backend is unreachable, the
// only honest thing to show is that fact (see docs/BACKEND_API.md and the project's "no fabricated scores" rule). All
// status comes from GET /system/health. The online state is the quiet chip in the TopBar; only the offline state
// needs a full-width notice.
export function BackendStatusBanner() {
  const { backendOnline } = useSession()

  if (backendOnline !== false) return null

  return (
    <div role="alert" className="flex items-center justify-center gap-2 border-b border-warning/30 bg-warning/10 px-4 py-2 text-center">
      <ShieldAlert className="h-3.5 w-3.5 shrink-0 text-warning" strokeWidth={1.5} aria-hidden />
      <span className="text-xs text-warning">
        {import.meta.env.DEV
          ? 'Backend unreachable - start it with `uvicorn backend.main:app --reload`'
          : 'Backend temporarily unreachable - please try again shortly'}
      </span>
    </div>
  )
}
