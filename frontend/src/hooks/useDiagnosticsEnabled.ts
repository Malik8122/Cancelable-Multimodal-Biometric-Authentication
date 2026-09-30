import { useSession } from '../context/SessionContext'

/** True when the backend runs in development/debug mode (DEBUG_SCORES) - the only time diagnostics are shown. */
export function useDiagnosticsEnabled(): boolean {
  return !!useSession().health?.diagnostics_enabled
}
