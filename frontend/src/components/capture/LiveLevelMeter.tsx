import { useEffect, useRef } from 'react'
import { cn } from '../../lib/utils'

const BARS = 32
const SAMPLE_MS = 70

/**
 * The real microphone input level, scrolling right-to-left while recording (one bar per ~70 ms). Heights are written
 * straight to the DOM from requestAnimationFrame - no React state per frame. When nothing is being recorded the bars
 * are flat: nothing here is simulated.
 */
export function LiveLevelMeter({ getAnalyser, active, tone }: { getAnalyser: () => AnalyserNode | null; active: boolean; tone: 'idle' | 'live' | 'success' | 'warning' }) {
  const barsRef = useRef<(HTMLSpanElement | null)[]>([])

  useEffect(() => {
    const levels = new Array<number>(BARS).fill(0)
    const paint = () => levels.forEach((level, i) => {
      const bar = barsRef.current[i]
      if (bar) bar.style.transform = `scaleY(${Math.max(0.06, level)})`
    })
    if (!active) {
      paint()
      return
    }
    let raf = 0
    let last = 0
    let buffer: Uint8Array<ArrayBuffer> | null = null
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick)
      if (now - last < SAMPLE_MS) return
      last = now
      const analyser = getAnalyser()
      if (!analyser) return
      if (!buffer || buffer.length !== analyser.fftSize) buffer = new Uint8Array(analyser.fftSize)
      analyser.getByteTimeDomainData(buffer)
      let sum = 0
      for (const v of buffer) sum += ((v - 128) / 128) ** 2
      const rms = Math.sqrt(sum / buffer.length)
      // Perceptual scaling: speech RMS is small, so map roughly -48..0 dBFS onto the bar height.
      const db = 20 * Math.log10(Math.max(rms, 1e-5))
      levels.shift()
      levels.push(Math.min(1, Math.max(0, (db + 48) / 48)))
      paint()
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [active, getAnalyser])

  const color = tone === 'live' ? 'bg-info' : tone === 'success' ? 'bg-success/60' : tone === 'warning' ? 'bg-warning/60' : 'bg-foreground/20'

  return (
    <div
      role="img"
      aria-label={active ? 'Live microphone input level' : 'Microphone idle'}
      className={cn('flex h-16 items-center justify-center gap-[3px] rounded-xl border bg-black/40 px-3 transition-colors duration-300', active ? 'border-info/40' : 'border-border')}
    >
      {Array.from({ length: BARS }, (_, i) => (
        <span
          key={i}
          ref={(el) => {
            barsRef.current[i] = el
          }}
          className={cn('h-12 w-1 origin-center rounded-full transition-[transform,background-color] duration-100', color)}
          style={{ transform: 'scaleY(0.06)' }}
        />
      ))}
    </div>
  )
}
