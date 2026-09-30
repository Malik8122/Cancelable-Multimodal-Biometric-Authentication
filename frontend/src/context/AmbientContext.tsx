import { useState, type ReactNode } from 'react'
import { AmbientContext, type AmbientState } from './ambientState'

export function AmbientProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AmbientState>('idle')
  return <AmbientContext.Provider value={{ state, setState }}>{children}</AmbientContext.Provider>
}
