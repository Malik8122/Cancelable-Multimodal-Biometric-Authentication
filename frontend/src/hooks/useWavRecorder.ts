import { useCallback, useRef, useState } from 'react'
import { MICROPHONE_CONSTRAINTS } from '../config/voice'
import { encodeWav } from '../utils/wav'

export type RecorderState = 'idle' | 'recording' | 'processing' | 'error'

export function useWavRecorder() {
  const [state, setState] = useState<RecorderState>('idle')
  const [error, setError] = useState<string | null>(null)
  const contextRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  // Display only: a passive analyser on a parallel branch, read by the live level meter. It is not part of the
  // recording chain (source -> processor), so the recorded samples are exactly what they were without it.
  const analyserRef = useRef<AnalyserNode | null>(null)
  const chunksRef = useRef<Float32Array[]>([])

  const start = useCallback(async () => {
    setError(null)
    try {
      // The same explicit constraints for enrollment and verification (config/voice.ts).
      const stream = await navigator.mediaDevices.getUserMedia({ audio: MICROPHONE_CONSTRAINTS })
      streamRef.current = stream
      const context = new AudioContext()
      contextRef.current = context
      const source = context.createMediaStreamSource(stream)
      sourceRef.current = source
      const processor = context.createScriptProcessor(4096, 1, 1)
      processorRef.current = processor
      chunksRef.current = []

      processor.onaudioprocess = (event) => {
        chunksRef.current.push(new Float32Array(event.inputBuffer.getChannelData(0)))
      }
      source.connect(processor)
      const analyser = context.createAnalyser()
      analyser.fftSize = 1024
      source.connect(analyser)
      analyserRef.current = analyser
      processor.connect(context.destination)
      setState('recording')
    } catch (err) {
      setState('error')
      setError(err instanceof Error ? err.message : 'Microphone access was denied.')
    }
  }, [])

  const stop = useCallback(async (): Promise<Blob | null> => {
    setState('processing')
    const context = contextRef.current
    processorRef.current?.disconnect()
    analyserRef.current?.disconnect()
    analyserRef.current = null
    sourceRef.current?.disconnect()
    streamRef.current?.getTracks().forEach((track) => track.stop())

    if (!context) {
      setState('idle')
      return null
    }

    const totalLength = chunksRef.current.reduce((sum, chunk) => sum + chunk.length, 0)
    const merged = new Float32Array(totalLength)
    let position = 0
    for (const chunk of chunksRef.current) {
      merged.set(chunk, position)
      position += chunk.length
    }
    const sampleRate = context.sampleRate
    await context.close()

    setState('idle')
    if (totalLength === 0) return null
    return encodeWav(merged, sampleRate)
  }, [])

  /** The live input analyser while recording (null otherwise) - for level display only. */
  const getAnalyser = useCallback(() => analyserRef.current, [])

  return { state, error, start, stop, getAnalyser }
}
