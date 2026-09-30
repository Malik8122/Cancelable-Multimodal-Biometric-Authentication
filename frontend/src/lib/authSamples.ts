// Assembles the /authenticate/fusion request from the captured factors - one face capture, the voice sentences (two),
// one hand gesture sample. Pure, so the protocol tests can check it directly.

import type { FusionSample } from '../api/client'
import type { Modality } from '../api/types'

export interface CapturedSample {
  blob: Blob
  filename: string
}

export function buildFusionSamples(
  selected: readonly Modality[],
  captured: Partial<Record<Modality, CapturedSample | readonly CapturedSample[]>>,
): FusionSample[] {
  return selected.flatMap((modality) => {
    const entry = captured[modality]
    const samples: readonly CapturedSample[] = entry === undefined ? [] : Array.isArray(entry) ? entry : [entry as CapturedSample]
    if (samples.length === 0) throw new Error(`${modality} has not been captured`)
    return samples.map((sample) => ({ modality, sample: sample.blob, filename: sample.filename }))
  })
}
