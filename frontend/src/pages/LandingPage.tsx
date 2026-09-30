import { AnimatePresence, motion, useMotionTemplate, useMotionValue, useSpring, useTransform } from 'motion/react'
import { ArrowDown, ArrowRight, LockKeyhole, ShieldCheck } from 'lucide-react'
import { useState, type MouseEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { BuildingSilhouette } from '../components/campus/BuildingSilhouette'
import type { Building } from '../config/buildings'
import { MODALITY_ICON } from '../config/modalityIcons'
import { useBuildings } from '../context/BuildingsContext'
import { useReducedMotion } from '../hooks/useReducedMotion'
import { EASE_OUT, rise, stagger } from '../lib/motion'

const SIGNALS = [
  { key: 'face', label: 'Face', y: 40 },
  { key: 'voice', label: 'Voice', y: 100 },
  { key: 'hand', label: 'Hand gesture', y: 160 },
] as const

// The system in one picture: three independent biometric signals converge into one protected identity decision.
// Drawn once on load (no loop).
function SignalConvergence() {
  return (
    <div className="relative mx-auto aspect-[16/10] w-full max-w-md">
      <svg viewBox="0 0 320 200" className="absolute inset-0 h-full w-full" aria-hidden>
        <defs>
          <linearGradient id="signal" x1="0" x2="1">
            <stop offset="0%" stopColor="var(--color-info)" stopOpacity="0.9" />
            <stop offset="100%" stopColor="var(--color-primary)" stopOpacity="0.9" />
          </linearGradient>
        </defs>
        {SIGNALS.map((s, i) => (
          <motion.path
            key={s.key}
            d={`M 128 ${s.y} C 185 ${s.y}, 190 100, 232 100`}
            fill="none"
            stroke="url(#signal)"
            strokeWidth={1.5}
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 1.1, delay: 0.5 + i * 0.15, ease: 'easeInOut' }}
          />
        ))}
        {SIGNALS.map((s, i) => (
          <motion.circle
            key={`pulse-${s.key}`}
            r={2.5}
            className="fill-info"
            initial={{ offsetDistance: '0%', opacity: 0 }}
            animate={{ offsetDistance: '100%', opacity: [0, 1, 1, 0] }}
            transition={{ duration: 1.1, delay: 1.5 + i * 0.15, ease: 'easeInOut' }}
            style={{ offsetPath: `path("M 128 ${s.y} C 185 ${s.y}, 190 100, 232 100")` }}
          />
        ))}
      </svg>
      {SIGNALS.map((s, i) => {
        const Icon = MODALITY_ICON[s.key]
        return (
          <motion.div
            key={s.key}
            className="absolute flex -translate-y-1/2 items-center gap-2"
            style={{ left: '2%', top: `${(s.y / 200) * 100}%` }}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 + i * 0.12, duration: 0.4, ease: EASE_OUT }}
          >
            <span className="flex h-10 w-10 items-center justify-center rounded-xl border border-info/40 bg-info/10 text-info sm:h-11 sm:w-11">
              <Icon className="h-5 w-5" strokeWidth={1.5} aria-hidden />
            </span>
            <span className="hidden text-xs font-medium text-foreground sm:inline">{s.label}</span>
          </motion.div>
        )
      })}
      <motion.div
        className="absolute flex -translate-x-1/2 -translate-y-1/2 flex-col items-center gap-2"
        style={{ left: '80%', top: '50%' }}
        initial={{ opacity: 0, scale: 0.7 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 1.45, type: 'spring', stiffness: 220, damping: 18 }}
      >
        <span className="flex h-16 w-16 items-center justify-center rounded-2xl border border-primary/50 bg-primary/15 text-primary shadow-[0_0_40px_-8px_var(--color-primary)] sm:h-20 sm:w-20">
          <ShieldCheck className="h-8 w-8 sm:h-10 sm:w-10" strokeWidth={1.5} aria-hidden />
        </span>
        <span className="text-eyebrow whitespace-nowrap text-foreground">Secure identity</span>
      </motion.div>
    </div>
  )
}

// A facility card: a staggered entrance, a soft spotlight that follows the cursor, and a very subtle tilt (max ~4 deg).
function FacilityCard({ building, index, onEnter, disabled }: { building: Building; index: number; onEnter: () => void; disabled: boolean }) {
  const mx = useMotionValue(0.5)
  const my = useMotionValue(0.5)
  const rotateX = useSpring(useTransform(my, [0, 1], [4, -4]), { stiffness: 200, damping: 20 })
  const rotateY = useSpring(useTransform(mx, [0, 1], [-4, 4]), { stiffness: 200, damping: 20 })
  const spotX = useTransform(mx, (v) => `${v * 100}%`)
  const spotY = useTransform(my, (v) => `${v * 100}%`)
  const spotlight = useMotionTemplate`radial-gradient(260px circle at ${spotX} ${spotY}, color-mix(in srgb, var(--color-primary) 16%, transparent), transparent 70%)`
  const [hovered, setHovered] = useState(false)
  // Cursor-driven styles bypass MotionConfig, so the tilt is switched off here for reduced motion.
  const reducedMotion = useReducedMotion()

  const track = (e: MouseEvent<HTMLButtonElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    mx.set((e.clientX - rect.left) / rect.width)
    my.set((e.clientY - rect.top) / rect.height)
  }
  const leave = () => {
    mx.set(0.5)
    my.set(0.5)
    setHovered(false)
  }

  return (
    <motion.li variants={rise} style={{ perspective: 900 }}>
      <motion.button
        type="button"
        onClick={onEnter}
        disabled={disabled}
        onMouseMove={track}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={leave}
        onFocus={() => setHovered(true)}
        onBlur={() => setHovered(false)}
        style={reducedMotion ? undefined : { rotateX, rotateY }}
        whileTap={{ scale: 0.985 }}
        aria-label={`Enter ${building.name}, clearance level ${building.clearanceLevel}`}
        className="surface-card group relative flex h-full w-full cursor-pointer flex-col overflow-hidden p-4 text-left transition-colors duration-300 hover:border-primary/50 sm:p-5"
      >
        <motion.span aria-hidden className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{ background: spotlight }} />
        <div className="relative mb-4 flex h-28 items-end justify-center overflow-hidden rounded-xl border border-border bg-gradient-to-b from-[#0a1330] to-[#0e1b3a] sm:h-32">
          <BuildingSilhouette seed={index + 1} brightness={hovered ? 1 : 0.35} className="h-24 w-16 transition-all duration-500 sm:h-28 sm:w-20" />
        </div>
        <div className="relative flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className="text-card-title text-foreground">{building.name}</p>
            <p className="text-meta mt-0.5 line-clamp-2 text-muted-foreground">{building.description}</p>
          </div>
        </div>
        <div className="relative mt-4 flex items-center justify-between">
          <span className="rounded-full border border-border-strong px-2.5 py-0.5 text-[11px] font-medium text-muted-foreground">
            Clearance {building.clearanceLevel}
          </span>
          <span className="flex items-center gap-1 text-xs font-medium text-primary">
            Enter <ArrowRight className="h-3.5 w-3.5 transition-transform duration-200 group-hover:translate-x-0.5" strokeWidth={1.75} aria-hidden />
          </span>
        </div>
      </motion.button>
    </motion.li>
  )
}

export function LandingPage() {
  const navigate = useNavigate()
  const { buildings, status: buildingsStatus, error: buildingsError, reload } = useBuildings()
  const reducedMotion = useReducedMotion()
  const [entering, setEntering] = useState<Building | null>(null)

  const enterBuilding = (building: Building) => {
    if (entering) return
    setEntering(building)
    const delay = reducedMotion ? 250 : 1500
    window.setTimeout(() => navigate(`/building/${building.id}`), delay)
  }

  return (
    <div className="relative">
      {/* Hero */}
      <section className="mx-auto grid max-w-6xl items-center gap-10 px-4 pt-10 pb-12 sm:px-6 sm:pt-16 lg:grid-cols-[1.05fr_1fr] lg:gap-6">
        <motion.div variants={stagger} initial="hidden" animate="show" className="text-center lg:text-left">
          <motion.p variants={rise} className="text-eyebrow mb-3 text-info">
            Cancelable Multimodal Biometric Authentication
          </motion.p>
          <motion.h1 variants={rise} className="mb-4 font-heading text-3xl leading-tight font-semibold tracking-tight text-foreground sm:text-5xl">
            One identity.
            <br />
            <span className="text-muted-foreground">Three independent signals.</span>
          </motion.h1>
          <motion.p variants={rise} className="mx-auto mb-7 max-w-lg text-[15px] leading-relaxed text-muted-foreground lg:mx-0">
            Face, voice and a dynamic hand gesture are each verified against keyed, revocable templates - never raw images,
            recordings or video - and fused into a single access decision for critical infrastructure.
          </motion.p>
          <motion.div variants={rise} className="flex flex-col items-center gap-3 sm:flex-row sm:justify-center lg:justify-start">
            <a href="#facilities" className="btn btn-primary w-full px-6 sm:w-auto">
              Choose a facility
              <ArrowDown className="h-4 w-4" strokeWidth={1.75} aria-hidden />
            </a>
            <Link to="/template-protection" className="btn btn-secondary w-full px-6 sm:w-auto">
              <LockKeyhole className="h-4 w-4" strokeWidth={1.5} aria-hidden />
              How templates are protected
            </Link>
          </motion.div>
        </motion.div>
        <SignalConvergence />
      </section>

      {/* Facilities */}
      <section id="facilities" className="mx-auto max-w-6xl scroll-mt-20 px-4 pb-10 sm:px-6" aria-labelledby="facilities-title">
        <div className="mb-5 flex flex-wrap items-end justify-between gap-2">
          <div>
            <p className="text-eyebrow text-muted-foreground">National critical infrastructure campus</p>
            <h2 id="facilities-title" className="text-section-title text-foreground">
              Select a facility checkpoint
            </h2>
          </div>
          <p className="text-meta text-muted-foreground">Each building runs its own biometric identity checkpoint.</p>
        </div>

        {buildingsStatus !== 'ready' && (
          <div className="surface-card p-6 text-center text-sm text-muted-foreground">
            {buildingsStatus === 'loading' ? (
              'Loading facilities...'
            ) : (
              <>
                <p className="mb-3">Facilities could not be loaded: {buildingsError}</p>
                <button type="button" onClick={reload} className="btn btn-secondary text-xs">
                  Retry
                </button>
              </>
            )}
          </div>
        )}

        <motion.ul variants={stagger} initial="hidden" whileInView="show" viewport={{ once: true, margin: '-60px' }} className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {buildings.map((building, i) => (
            <FacilityCard key={building.id} building={building} index={i} onEnter={() => enterBuilding(building)} disabled={!!entering} />
          ))}
        </motion.ul>
      </section>

      {/* Entering a facility: the building grows toward the viewer, the doors part, and the lobby fades in. */}
      <AnimatePresence>
        {entering && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-background"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: reducedMotion ? 0.15 : 0.4 }}
          >
            {!reducedMotion && (
              <>
                <motion.div initial={{ scale: 0.7, opacity: 0.9 }} animate={{ scale: 2.6, opacity: 0 }} transition={{ duration: 1.3, ease: [0.4, 0, 0.2, 1] }}>
                  <BuildingSilhouette seed={buildings.indexOf(entering) + 1} brightness={1} className="h-64 w-36" />
                </motion.div>
                <motion.p
                  className="text-eyebrow absolute bottom-[22%] text-muted-foreground"
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  Entering {entering.name}
                </motion.p>
              </>
            )}
            <motion.div
              className="absolute inset-0 bg-background"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: reducedMotion ? 0 : 0.85 }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
