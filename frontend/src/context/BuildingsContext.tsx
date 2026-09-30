import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { getBuildings } from '../api/client'
import { ApiError } from '../api/types'
import { toBuilding, type Building } from '../config/buildings'

interface BuildingsValue {
  buildings: Building[]
  getBuilding: (id: string) => Building | undefined
  /** 'loading' until the first response; 'error' when the backend could not be reached. */
  status: 'loading' | 'ready' | 'error'
  error: string | null
  reload: () => void
}

const BuildingsContext = createContext<BuildingsValue | null>(null)

// Loads the buildings once from the backend (GET /buildings): id, name, clearance level, description. Buildings are
// context only - they say nothing about which biometric factors to use.
export function BuildingsProvider({ children }: { children: ReactNode }) {
  const [buildings, setBuildings] = useState<Building[]>([])
  const [status, setStatus] = useState<BuildingsValue['status']>('loading')
  const [error, setError] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let cancelled = false
    getBuildings()
      .then((policies) => {
        if (cancelled) return
        setBuildings(policies.map(toBuilding))
        setStatus('ready')
        setError(null)
      })
      .catch((err) => {
        if (cancelled) return
        setStatus('error')
        setError(err instanceof ApiError ? err.detail : 'The backend is unreachable.')
      })
    return () => {
      cancelled = true
    }
  }, [attempt])

  const reload = useCallback(() => {
    setStatus('loading')
    setAttempt((n) => n + 1)
  }, [])

  const value = useMemo<BuildingsValue>(
    () => ({ buildings, getBuilding: (id) => buildings.find((b) => b.id === id), status, error, reload }),
    [buildings, status, error, reload],
  )
  return <BuildingsContext.Provider value={value}>{children}</BuildingsContext.Provider>
}

export function useBuildings(): BuildingsValue {
  const ctx = useContext(BuildingsContext)
  if (!ctx) throw new Error('useBuildings must be used within BuildingsProvider')
  return ctx
}
