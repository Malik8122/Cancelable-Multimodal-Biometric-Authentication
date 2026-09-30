import { createContext, useContext, type ReactNode } from 'react'
import { useAuthSession } from '../hooks/useAuthSession'

type SessionValue = ReturnType<typeof useAuthSession>

const SessionContext = createContext<SessionValue | null>(null)

export function SessionProvider({ children }: { children: ReactNode }) {
  const session = useAuthSession()
  return <SessionContext.Provider value={session}>{children}</SessionContext.Provider>
}

export function useSession(): SessionValue {
  const ctx = useContext(SessionContext)
  if (!ctx) throw new Error('useSession must be used within SessionProvider')
  return ctx
}
