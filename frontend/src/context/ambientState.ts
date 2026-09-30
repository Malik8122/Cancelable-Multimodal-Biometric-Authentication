import { createContext } from 'react'

/**
 * What the console is doing right now, reflected (very subtly) by the app backdrop's light. Pages report it; nothing
 * here is inferred. idle = waiting for the user, capturing = a sensor is open, processing = the backend is verifying,
 * success / failure = the last decision.
 */
export type AmbientState = 'idle' | 'capturing' | 'processing' | 'success' | 'failure'

export const AmbientContext = createContext<{ state: AmbientState; setState: (s: AmbientState) => void }>({
  state: 'idle',
  setState: () => {},
})
