import type { BuildingInfo, Modality } from '../api/types'

// A building is authentication CONTEXT only (id, name, clearance level, description) and defines no biometric policy.
// The user decides which modalities to enroll and which to present in a session. The list itself comes from the
// backend (config/buildings.json via GET /buildings); this module adapts it and holds pure display helpers.
export interface Building {
  id: string
  name: string
  clearanceLevel: 'III' | 'IV' | 'V'
  description: string
}

export function toBuilding(info: BuildingInfo): Building {
  return {
    id: info.id,
    name: info.name,
    clearanceLevel: info.clearance_level as Building['clearanceLevel'],
    description: info.description,
  }
}

export const MODALITY_LABEL: Record<Modality, string> = {
  face: 'Face',
  voice: 'Voice',
  hand: 'Hand Gesture',
}

/** The biometric factors a user can enroll and present. */
export const FACTORS: Modality[] = ['face', 'voice', 'hand']

export function joinModalities(modalities: Modality[]): string {
  return modalities.map((m) => MODALITY_LABEL[m]).join(' + ')
}
