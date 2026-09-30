import { motion, useScroll, useSpring } from 'motion/react'
import { Database, Fingerprint, Gauge, Grid3x3, KeyRound, Layers, Lock, RefreshCw, ShieldCheck, Shuffle, Sparkles } from 'lucide-react'
import { useRef, useState } from 'react'
import { MODALITY_ICON } from '../config/modalityIcons'
import type { Modality, TemplateSetStatus } from '../api/types'
import { TemplateSetCard } from '../components/biometric/TemplateSetCard'

const POOL_SIZE = 4
const DEMO_MODALITIES: Modality[] = ['face', 'voice', 'hand']

// The core research idea, stage by stage - the pipeline this backend actually runs. Per-modality specifics are stated
// where they differ (face / voice: BioHash; hand gesture: keyed orthonormal transform of the feature sequence).
const PIPELINE = [
  {
    key: 'input',
    title: 'Biometric input',
    desc: 'A face image, voice recordings, and a hand-landmark sequence (MediaPipe runs in the browser - only landmark coordinates are sent, never video).',
    icon: Fingerprint,
  },
  {
    key: 'embed',
    title: 'Embedding / features',
    desc: 'Face: 512-d InceptionResnetV1 embedding. Voice: 192-d ECAPA-TDNN embedding. Hand: a normalized 64-frame x 12-feature trajectory and hand-shape sequence. They exist only in memory and are discarded after the templates are built.',
    icon: Sparkles,
  },
  {
    key: 'hkdf',
    title: 'Key derivation',
    desc: 'HKDF-SHA256 derives an independent key per template set, modality, application and key version from the master secret. Keys are derived on demand, never stored.',
    icon: KeyRound,
  },
  {
    key: 'transform',
    title: 'Template transformation',
    desc: 'Face and voice: BioHash - a keyed random orthonormal projection, keyed quantization and a bit permutation (lossy, 256 bits). Hand gesture: a keyed orthonormal transform of each enrolled sequence, which preserves DTW distances; it is revocable but invertible by anyone holding the key.',
    icon: Shuffle,
  },
  {
    key: 'protected',
    title: 'Protected template',
    desc: 'Only the transformed templates and lifecycle metadata are stored - one complete credential per template set. No image, audio, video or embedding is ever written.',
    icon: Database,
  },
]

// What happens at authentication time, after the protected templates exist.
const RUNTIME = [
  { key: 'pool', title: 'Template set pool', desc: 'Set 1 is ACTIVE; Sets 2-4 are STANDBY. A set is one complete multimodal credential, never mixed with another set.', icon: Grid3x3 },
  { key: 'auth', title: 'Authentication', desc: "A fresh capture is transformed under the ACTIVE set's keys and compared with the ACTIVE set's templates only.", icon: Lock },
  { key: 'fusion', title: 'Fusion', desc: 'Per-modality results are fused into one decision under the fusion policy (ALL_REQUIRED by default).', icon: Layers },
  { key: 'decision', title: 'Access decision', desc: 'The client receives the decision, the verified factors and the template set version - per-factor scores only in debug mode.', icon: Gauge },
]

const MODALITIES = [
  { key: 'face', label: 'Face' },
  { key: 'voice', label: 'Voice' },
  { key: 'hand', label: 'Hand gesture' },
] as const

interface DemoSet {
  version: number
  status: TemplateSetStatus
}

const freshPool = (): DemoSet[] =>
  Array.from({ length: POOL_SIZE }, (_, i) => ({ version: i + 1, status: i === 0 ? 'ACTIVE' : 'STANDBY' }))

// The scroll-driven pipeline: a line draws down the stages as you scroll, and each stage lights up as it arrives.
function ScrollPipeline() {
  const ref = useRef<HTMLOListElement>(null)
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start 0.75', 'end 0.55'] })
  const progress = useSpring(scrollYProgress, { stiffness: 120, damping: 24 })
  return (
    <ol ref={ref} className="relative mb-16 space-y-5 pl-14 sm:pl-16">
      <span aria-hidden className="absolute top-2 bottom-2 left-[1.35rem] w-px bg-border-strong sm:left-[1.6rem]" />
      <motion.span aria-hidden className="absolute top-2 bottom-2 left-[1.35rem] w-px origin-top bg-gradient-to-b from-info via-primary to-success sm:left-[1.6rem]" style={{ scaleY: progress }} />
      {PIPELINE.map((stage, i) => (
        <motion.li
          key={stage.key}
          initial={{ opacity: 0.25, x: 12 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: false, margin: '-35% 0px -35% 0px' }}
          transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
          className="relative"
        >
          <span className="absolute top-3 -left-14 flex h-11 w-11 items-center justify-center rounded-xl border border-primary/40 bg-background text-primary sm:-left-16 sm:h-12 sm:w-12">
            <stage.icon className="h-5 w-5" strokeWidth={1.5} aria-hidden />
          </span>
          <div className="surface-card p-4 sm:p-5">
            <p className="text-eyebrow text-muted-foreground">Stage {i + 1}</p>
            <h3 className="text-card-title mb-1 text-foreground">{stage.title}</h3>
            <p className="text-sm leading-relaxed text-muted-foreground">{stage.desc}</p>
          </div>
        </motion.li>
      ))}
    </ol>
  )
}

export function TemplateProtectionPage() {
  const [pool, setPool] = useState<DemoSet[]>(freshPool)
  const [flash, setFlash] = useState<number | null>(null)

  const standbyLeft = pool.filter((s) => s.status === 'STANDBY').length
  const active = pool.find((s) => s.status === 'ACTIVE')

  const revoke = () => {
    if (!active || standbyLeft === 0) return
    const next = pool.find((s) => s.status === 'STANDBY')!
    setFlash(next.version)
    setPool((prev) =>
      prev.map((s) =>
        s.version === active.version ? { ...s, status: 'REVOKED' } : s.version === next.version ? { ...s, status: 'ACTIVE' } : s,
      ),
    )
    window.setTimeout(() => setFlash(null), 900)
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-10 sm:px-6 sm:py-14">
      <header className="mb-10 text-center">
        <p className="text-eyebrow mb-2 text-info">Research explainer</p>
        <h1 className="text-page-title mb-3 text-foreground">Cancelable Template Protection</h1>
        <p className="mx-auto max-w-xl text-sm leading-relaxed text-muted-foreground">
          How enrollment turns each biometric into keyed, revocable templates - and why a revoked credential can be replaced
          without re-enrolling. Scroll to follow the pipeline this backend runs.
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          {MODALITIES.map((m) => {
            const Icon = MODALITY_ICON[m.key]
            return (
              <span key={m.key} className="flex items-center gap-1.5 rounded-full border border-primary/25 bg-primary/10 px-3.5 py-1.5 text-xs font-medium text-primary">
                <Icon className="h-3.5 w-3.5" strokeWidth={1.5} aria-hidden />
                {m.label}
              </span>
            )
          })}
        </div>
      </header>

      <ScrollPipeline />

      <section className="mb-16" aria-labelledby="runtime-title">
        <h2 id="runtime-title" className="text-section-title mb-4 text-foreground">
          At authentication time
        </h2>
        <motion.ul
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: '-60px' }}
          variants={{ hidden: {}, show: { transition: { staggerChildren: 0.08 } } }}
          className="grid grid-cols-1 gap-3 sm:grid-cols-2"
        >
          {RUNTIME.map((stage) => (
            <motion.li
              key={stage.key}
              variants={{ hidden: { opacity: 0, y: 10 }, show: { opacity: 1, y: 0 } }}
              className="surface-card flex gap-3 p-4"
            >
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-primary/25 bg-primary/10 text-primary">
                <stage.icon className="h-4 w-4" strokeWidth={1.5} aria-hidden />
              </span>
              <span>
                <span className="text-card-title block text-foreground">{stage.title}</span>
                <span className="text-sm text-muted-foreground">{stage.desc}</span>
              </span>
            </motion.li>
          ))}
        </motion.ul>
      </section>

      {/* Revocation animation */}
      <section className="surface-card p-5 sm:p-8" aria-labelledby="revocation-title">
        <p className="text-eyebrow mb-1 text-primary">Interactive demo</p>
        <h2 id="revocation-title" className="text-section-title mb-2 text-foreground">Template set revocation</h2>
        <p className="mb-6 text-sm text-muted-foreground">
          Revoking the active set retires its face, voice and hand gesture templates together and activates the oldest
          standby set for all of them at once - the same person keeps authenticating, nothing is re-enrolled. With no
          standby set left, re-enrollment is required.
        </p>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {pool.map((s) => (
            <motion.div
              key={s.version}
              animate={flash === s.version ? { scale: [1, 1.05, 1] } : { scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              <TemplateSetCard set={{ version: s.version, status: s.status, modalities: DEMO_MODALITIES }} />
            </motion.div>
          ))}
        </div>

        <button
          onClick={revoke}
          disabled={!active || standbyLeft === 0}
          type="button"
          className="btn btn-primary mt-6 w-full"
        >
          <ShieldCheck className="h-4 w-4" strokeWidth={1.5} />
          {standbyLeft === 0 ? 'Template set pool exhausted - re-enrollment required' : 'Revoke Active Set'}
        </button>
        {standbyLeft === 0 && (
          <button
            onClick={() => setPool(freshPool())}
            type="button"
            className="btn btn-secondary mt-3 w-full text-muted-foreground"
          >
            <RefreshCw className="h-4 w-4" strokeWidth={1.5} />
            Re-enroll (reset demo)
          </button>
        )}
        <p className="mt-3 text-center text-meta text-muted-foreground">
          Illustrative demo - mirrors POST /revoke-template (which needs biometric authorization and answers 409 when the
          set pool is exhausted)
        </p>
      </section>
    </div>
  )
}
