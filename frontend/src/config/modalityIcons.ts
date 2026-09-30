import { Hand, Mic, ScanFace, type LucideIcon } from 'lucide-react'
import type { Modality } from '../api/types'

/** One icon per biometric factor, used everywhere a modality is named. */
export const MODALITY_ICON: Record<Modality, LucideIcon> = { face: ScanFace, voice: Mic, hand: Hand }
