import { motion } from 'motion/react'
import { Check } from 'lucide-react'
import { useEffect, useState } from 'react'
import { emptyVoiceSlots, voicePrompts } from '../../config/protocol'
import { quick } from '../../lib/motion'
import { VoiceCapture } from './VoiceCapture'

export interface VoiceRecording {
  blob: Blob
  filename: string
}

/** One slot per prompt of the current protocol, each filled only once its recording passed the quality check. */
export type VoiceSentences = (VoiceRecording | null)[]


// One recorder per prompt of the protocol (config/protocol.ts): ENROLLMENT records every enrollment prompt
// ("Sentence 1 / N" ...), VERIFICATION records the authentication prompts (two). Each recording is checked and
// retried on its own, so a valid recording never has to be redone because another one failed.
export function VoiceSentencesCapture({ mode, onChange }: { mode: 'register' | 'verify'; onChange: (sentences: VoiceSentences) => void }) {
  const prompts = voicePrompts(mode)
  const [sentences, setSentences] = useState<VoiceSentences>(() => emptyVoiceSlots<VoiceRecording>(mode))

  useEffect(() => {
    onChange(sentences)
  }, [sentences, onChange])

  const set = (index: number, value: VoiceRecording | null) => setSentences((prev) => prev.map((s, i) => (i === index ? value : s)))

  const accepted = sentences.filter(Boolean).length
  const total = prompts.length
  const ready = accepted === total

  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-3">
        <p className="text-meta text-muted-foreground">
          {mode === 'register'
            ? `Enrollment records ${total} sentences, each checked for recording quality.`
            : `Verification records ${total} sentences, each checked for recording quality. Speak each one clearly.`}
        </p>
        <p className={`flex shrink-0 items-center gap-1.5 text-xs font-medium tabular-nums ${ready ? 'text-success' : 'text-muted-foreground'}`}>
          {ready && <Check className="h-3.5 w-3.5" strokeWidth={2.25} aria-hidden />}
          {total > 1 ? `${accepted} / ${total} accepted` : ready ? 'Voice captured' : 'Not recorded yet'}
        </p>
      </div>
      {total > 1 && (
        <div className="mb-4 h-1 overflow-hidden rounded-full bg-raised" aria-hidden>
          <motion.div className="h-full rounded-full bg-success/80" initial={false} animate={{ width: `${(accepted / total) * 100}%` }} transition={quick} />
        </div>
      )}
      <div className={`grid grid-cols-1 gap-4 ${total > 1 ? 'md:grid-cols-2' : 'mx-auto max-w-xl'}`}>
        {prompts.map((prompt, index) => (
          <VoiceCapture
            key={index}
            mode={mode}
            sentence={index + 1}
            total={total}
            prompt={prompt}
            onCapture={(blob, filename) => set(index, { blob, filename })}
            onReset={() => set(index, null)}
          />
        ))}
      </div>
    </div>
  )
}
