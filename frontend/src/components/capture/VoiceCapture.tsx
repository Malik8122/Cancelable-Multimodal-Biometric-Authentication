import { AnimatePresence, motion } from 'motion/react'
import { AlertCircle, Check, Mic, MicOff, RotateCcw, Square, Upload } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { checkVoiceQuality } from '../../api/client'
import type { VoiceQualityCheckResponse } from '../../api/types'
import { VOICE_AUTHENTICATION_PROMPT, VOICE_MAX_SECONDS, VOICE_MIN_SECONDS } from '../../config/voice'
import { useWavRecorder } from '../../hooks/useWavRecorder'
import { rise } from '../../lib/motion'
import { CaptureCard, StepCount } from '../biometric/CaptureCard'
import { CaptureStatus, type StatusTone } from '../biometric/CaptureStatus'
import { LiveLevelMeter } from './LiveLevelMeter'
import { Voiceprint } from './Voiceprint'
import { VoiceDiagnostics } from '../diagnostics/VoiceDiagnostics'
import { voiceprintLevels } from '../../lib/biometricVisuals'

interface Props {
  mode: 'register' | 'verify'
  /** Called only with a recording that passed the backend's capture-quality check. */
  onCapture: (blob: Blob, filename: string) => void
  /** Called when a previously passed recording is discarded (re-record) or a new one fails the check. */
  onReset?: () => void
  disabled?: boolean
  /** Card heading override, e.g. "Sentence 1". */
  title?: string
  /** The sentence to speak (defaults to the single authentication prompt of config/protocol.ts). */
  prompt?: string
  /** 1-based sentence number, used in the retry button and sent with the quality check. */
  sentence?: number
  /** How many sentences this capture step has in total (shown as "Sentence 1 / 2"). */
  total?: number
  /** A recording shorter than this is discarded; one reaching `maxSeconds` stops by itself. */
  minSeconds?: number
  maxSeconds?: number
}


type Check = { state: 'checking' } | { state: 'done'; result: VoiceQualityCheckResponse } | { state: 'unavailable' }

export function VoiceCapture({
  mode,
  onCapture,
  onReset,
  disabled,
  title,
  prompt = VOICE_AUTHENTICATION_PROMPT,
  sentence,
  total,
  minSeconds = VOICE_MIN_SECONDS,
  maxSeconds = VOICE_MAX_SECONDS,
}: Props) {
  const { state, error, start, stop, getAnalyser } = useWavRecorder()
  const [seconds, setSeconds] = useState(0)
  const [check, setCheck] = useState<Check | null>(null)
  const [tooShort, setTooShort] = useState(false)
  // Voiceprint of the accepted recording (its real loudness envelope), shown once the sentence passes.
  const [print, setPrint] = useState<number[] | null>(null)
  // Size of the last checked recording (development diagnostics only).
  const [fileBytes, setFileBytes] = useState<number | undefined>(undefined)
  const secondsRef = useRef(0)
  const timerRef = useRef<number | null>(null)

  const isRecording = state === 'recording'
  const passed = check?.state === 'unavailable' || (check?.state === 'done' && check.result.passed)
  const failed = check?.state === 'done' && !check.result.passed

  // The backend's gate decides; the UI only reports it. A failed recording is never handed to the parent.
  const evaluate = async (blob: Blob, filename: string) => {
    setCheck({ state: 'checking' })
    setFileBytes(blob.size)
    try {
      const result = await checkVoiceQuality(blob, filename, sentence)
      setCheck({ state: 'done', result })
      if (result.passed) {
        onCapture(blob, filename)
        void voiceprintLevels(blob).then(setPrint)
      } else onReset?.()
    } catch {
      // The check endpoint is unreachable: keep the recording - the backend re-checks it when it is submitted.
      setCheck({ state: 'unavailable' })
      onCapture(blob, filename)
      void voiceprintLevels(blob).then(setPrint)
    }
  }

  const handleStart = async () => {
    setSeconds(0)
    secondsRef.current = 0
    setCheck(null)
    setTooShort(false)
    setPrint(null)
    onReset?.()
    await start()
    timerRef.current = window.setInterval(() => {
      secondsRef.current += 1
      setSeconds((s) => s + 1)
    }, 1000)
  }

  const handleStop = async () => {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    const blob = await stop()
    if (!blob) return
    if (secondsRef.current < minSeconds) {
      setTooShort(true) // too short to be a usable voice sample: discard it
      return
    }
    await evaluate(blob, `voice-sentence-${sentence ?? 1}.wav`)
  }

  // Stop by itself at the maximum duration.
  const stopRef = useRef(handleStop)
  stopRef.current = handleStop
  useEffect(() => {
    if (isRecording && seconds >= maxSeconds) void stopRef.current()
  }, [isRecording, seconds, maxSeconds])

  const handleFile = (file: File) => {
    setTooShort(false)
    setPrint(null)
    void evaluate(file, file.name)
  }

  const reRecord = () => {
    setCheck(null)
    setPrint(null)
    onReset?.()
  }

  const heading = title ?? (mode === 'verify' ? 'Speak the displayed sentence' : sentence ? `Speak sentence ${sentence}` : 'Record your voice')

  const status: { tone: StatusTone; label: string } = error
    ? { tone: 'danger', label: 'Microphone unavailable' }
    : isRecording
      ? { tone: 'live', label: 'Listening' }
      : state === 'processing'
        ? { tone: 'processing', label: 'Processing recording...' }
        : check?.state === 'checking'
          ? { tone: 'processing', label: 'Checking recording quality...' }
          : tooShort
            ? { tone: 'warning', label: 'Too short - record again' }
            : failed
              ? { tone: 'warning', label: 'Re-record needed' }
              : check?.state === 'unavailable'
                ? { tone: 'idle', label: 'Recorded - checked on submit' }
                : passed
                  ? { tone: 'success', label: mode === 'verify' ? 'Voice captured' : 'Sentence accepted' }
                  : { tone: 'idle', label: 'Ready to record' }

  return (
    <CaptureCard
      modality="voice"
      eyebrow={mode === 'register' ? 'Voice enrollment' : 'Voice verification'}
      title={heading}
      state={isRecording ? 'active' : passed ? 'complete' : failed || tooShort || error ? 'attention' : 'idle'}
      aside={sentence && total ? <StepCount current={sentence} total={total} label="Sentence" /> : undefined}
    >
      <div className="mb-3 rounded-xl border border-border bg-raised/50 px-3.5 py-2.5">
        <p className="text-meta text-muted-foreground">Say this sentence</p>
        <p className="text-[15px] leading-snug text-foreground">&ldquo;{prompt}&rdquo;</p>
      </div>

      {/* Live input level while recording; once the sentence is accepted the meter folds away and the recording's
          voiceprint is printed in its place. */}
      <AnimatePresence mode="wait" initial={false}>
        {passed && print ? (
          <motion.div key="print" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <Voiceprint levels={print} label={`Voiceprint of sentence ${sentence ?? 1}: loudness of the accepted recording over time`} />
          </motion.div>
        ) : (
          <motion.div
            key="live"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1, scaleY: 1 }}
            exit={{ opacity: 0, scaleY: 0.15, transition: { duration: 0.22, ease: [0.4, 0, 1, 1] } }}
            className="origin-center"
          >
            <LiveLevelMeter getAnalyser={getAnalyser} active={isRecording} tone={isRecording ? 'live' : passed ? 'success' : failed ? 'warning' : 'idle'} />
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mt-3 mb-3 flex min-h-7 items-center justify-between gap-2">
        <CaptureStatus tone={status.tone} label={status.label} />
        {isRecording && (
          <span className="font-mono text-xs tabular-nums text-muted-foreground" aria-label={`${seconds} of ${maxSeconds} seconds`}>
            {seconds}s / {maxSeconds}s
          </span>
        )}
        {passed && print && <span className="text-xs text-muted-foreground">Voiceprint</span>}
      </div>

      <AnimatePresence initial={false} mode="popLayout">
        {error && (
          <motion.p key="mic" variants={rise} initial="hidden" animate="show" exit="exit" className="mb-3 flex items-start gap-2 rounded-lg border border-danger/30 bg-danger/10 p-2.5 text-xs text-danger">
            <MicOff className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.5} aria-hidden />
            <span>Microphone unavailable. Check microphone permissions and try again. ({error})</span>
          </motion.p>
        )}
        {tooShort && (
          <motion.p key="short" variants={rise} initial="hidden" animate="show" exit="exit" className="mb-3 text-xs text-warning">
            That recording was too short. Please record for at least {minSeconds} seconds.
          </motion.p>
        )}
        {check?.state === 'done' && (
          <motion.p
            key={check.result.passed ? 'passed' : 'failed'}
            variants={rise}
            initial="hidden"
            animate="show"
            exit="exit"
            className={`mb-3 flex items-start gap-2 rounded-lg border p-2.5 text-xs ${
              check.result.passed ? 'border-success/30 bg-success/10 text-success' : 'border-warning/30 bg-warning/10 text-warning'
            }`}
          >
            {check.result.passed ? (
              <Check className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={2} aria-hidden />
            ) : (
              <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.5} aria-hidden />
            )}
            {check.result.message}
          </motion.p>
        )}
        {check?.state === 'done' && check.result.diagnostics && (
          <motion.div key="diagnostics" variants={rise} initial="hidden" animate="show" exit="exit">
            <VoiceDiagnostics result={check.result} fileBytes={fileBytes} />
          </motion.div>
        )}
        {check?.state === 'unavailable' && (
          <motion.p key="unavailable" variants={rise} initial="hidden" animate="show" exit="exit" className="mb-3 text-xs text-muted-foreground">
            Quality could not be checked now; it is checked again when you submit.
          </motion.p>
        )}
      </AnimatePresence>

      {!check && !isRecording && !tooShort && (
        <p className="mb-4 text-[13px] leading-relaxed text-muted-foreground">
          Speak clearly for {minSeconds}-{maxSeconds} seconds, close to the microphone, in a quiet room. Recording stops
          automatically at {maxSeconds} seconds.
        </p>
      )}

      <div className="flex gap-2.5">
        {passed || failed ? (
          <button type="button" onClick={reRecord} className={`btn flex-1 ${failed ? 'btn-primary' : 'btn-secondary'}`}>
            <RotateCcw className="h-4 w-4" strokeWidth={1.5} aria-hidden />
            {failed ? `Retry ${sentence ? `sentence ${sentence}` : 'recording'}` : 'Re-record'}
          </button>
        ) : !isRecording ? (
          <motion.button
            type="button"
            whileTap={{ scale: 0.98 }}
            onClick={handleStart}
            disabled={disabled || state === 'processing' || check?.state === 'checking'}
            className="btn btn-primary flex-1"
          >
            <Mic className="h-4 w-4" strokeWidth={1.5} aria-hidden />
            {state === 'processing' ? 'Processing...' : 'Start recording'}
          </motion.button>
        ) : (
          <button type="button" onClick={handleStop} className="btn btn-danger flex-1">
            <Square className="h-4 w-4" strokeWidth={1.5} aria-hidden />
            Stop recording
          </button>
        )}
        <label className="btn btn-secondary text-muted-foreground has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-ring">
          <Upload className="h-4 w-4" strokeWidth={1.5} aria-hidden />
          <span className="hidden sm:inline">Upload WAV</span>
          <span className="sm:hidden">WAV</span>
          <input
            type="file"
            accept="audio/wav,audio/x-wav,audio/wave"
            className="sr-only"
            disabled={disabled || isRecording}
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
        </label>
      </div>
      <p className="mt-3 text-center text-meta text-muted-foreground">WAV, converted to 16 kHz mono on the server</p>
    </CaptureCard>
  )
}
