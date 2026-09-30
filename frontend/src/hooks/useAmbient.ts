import { useContext, useEffect } from 'react'
import { AmbientContext, type AmbientState } from '../context/ambientState'

export function useAmbientState(): AmbientState {
  return useContext(AmbientContext).state
}

/** Report the page's current state to the backdrop; it returns to idle when the page unmounts. */
export function useAmbient(state: AmbientState) {
  const { setState } = useContext(AmbientContext)
  useEffect(() => {
    setState(state)
  }, [state, setState])
  useEffect(() => () => setState('idle'), [setState])
}
