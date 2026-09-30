import { motion } from 'motion/react'
import { AlertCircle, ArrowRight, Hand, Layers, Loader2, Mic, ScanFace, ShieldCheck } from 'lucide-react'
import { useEffect } from 'react'
import { Link, useParams } from 'react-router-dom'
import type { Modality } from '../api/types'
import { ClearanceBadge } from '../components/biometric/ClearanceBadge'
import { ModalityBadge } from '../components/biometric/ModalityBadge'
import { EnrollmentStatusPill } from '../components/biometric/EnrollmentStatusPill'
import { FACTORS, MODALITY_LABEL } from '../config/buildings'
import { useBuildings } from '../context/BuildingsContext'
import { useSession } from '../context/SessionContext'
import { useEnrollmentProfile } from '../hooks/useEnrollmentProfile'
import { rise, stagger } from '../lib/motion'

// What presenting each factor involves at verification time (the authentication protocol: one sample each).
const FACTOR_DETAIL: Record<Modality, string> = {
  face: 'Camera - one capture',
  voice: 'Microphone - one sentence',
  hand: 'Camera - one gesture sample',
}

// Security lobby. The building is only the context of the session; WHICH biometric factors to use is the user's
// choice, from what they have enrolled. A facility prescribes no particular factor.
export function CheckpointPage() {
  const { buildingId } = useParams<{ buildingId: string }>()
  const { getBuilding, status: buildingsStatus } = useBuildings()
  const { userId, knownUsers, switchUser, userNames, rememberName } = useSession()
  const building = buildingId ? getBuilding(buildingId) : undefined
  const { statuses, enrolledList, displayName, hasDisplayName, error, loading } = useEnrollmentProfile(userId)

  // Keep the picker's local name cache in step with the backend (e.g. a user named on another device).
  useEffect(() => {
    if (hasDisplayName && displayName) rememberName(userId, displayName)
  }, [userId, displayName, hasDisplayName, rememberName])

  if (!building) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-16 text-center">
        <p className="text-muted-foreground">{buildingsStatus === 'loading' ? 'Loading facility...' : 'Unknown facility.'}</p>
        <Link to="/" className="mt-4 inline-block text-sm text-primary hover:underline">
          Return to the campus
        </Link>
      </div>
    )
  }

  const canAuthenticate = enrolledList.length > 0

  return (
    <div className="mx-auto max-w-5xl px-4 pt-8 pb-14 sm:px-6 sm:pt-12">
      <motion.header variants={stagger} initial="hidden" animate="show" className="mb-8 flex flex-col items-center gap-3 text-center">
        <motion.div variants={rise} className="flex flex-wrap items-center justify-center gap-2">
          <span className="text-eyebrow text-muted-foreground">{building.name}</span>
          <ClearanceBadge level={building.clearanceLevel} />
        </motion.div>
        <motion.h1 variants={rise} className="text-page-title text-foreground">
          Multimodal Biometric Authentication
        </motion.h1>
        <motion.p variants={rise} className="max-w-xl text-sm leading-relaxed text-muted-foreground">
          {building.description} Present any of your enrolled factors; they are verified against your protected templates
          and fused into one access decision.
        </motion.p>
      </motion.header>

      {/* Who is authenticating */}
      <motion.section variants={rise} initial="hidden" animate="show" className="surface-card mx-auto mb-5 flex max-w-3xl flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between sm:p-5">
        <div className="min-w-0">
          <p className="text-eyebrow text-muted-foreground">Registered user</p>
          {knownUsers.length > 1 ? (
            <>
              <label htmlFor="user-select" className="sr-only">
                Registered user
              </label>
              <select
                id="user-select"
                value={userId}
                onChange={(e) => switchUser(e.target.value)}
                className="mt-1 w-full min-w-56 rounded-lg border border-border-strong bg-raised px-3 py-2 text-sm text-foreground sm:w-auto"
              >
                {knownUsers.map((id) => (
                  <option key={id} value={id}>
                    {userNames[id] ?? (id === userId && displayName ? displayName : 'Unnamed user')}
                  </option>
                ))}
              </select>
            </>
          ) : (
            <p className="text-section-title mt-0.5 text-foreground">{displayName ?? (loading ? 'Loading...' : 'Unnamed user')}</p>
          )}
        </div>
        <p className="text-sm text-muted-foreground">
          {loading ? (
            <span className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} aria-hidden /> Checking enrollment...
            </span>
          ) : (
            <>
              <span className="font-semibold tabular-nums text-foreground">{enrolledList.length}</span> of {FACTORS.length} factors enrolled
            </>
          )}
        </p>
      </motion.section>

      {/* The three factors */}
      <motion.ul variants={stagger} initial="hidden" animate="show" className="mx-auto mb-5 grid max-w-3xl grid-cols-1 gap-3 sm:grid-cols-3" aria-label="Biometric factors">
        {FACTORS.map((modality) => {
          const status = statuses[modality]
          const isEnrolled = status === 'REGISTERED' || status === 'UPDATED'
          return (
            <motion.li key={modality} variants={rise} className={`surface-card flex flex-col gap-3 p-4 ${isEnrolled ? '' : 'border-dashed'}`}>
              <div className="flex items-center justify-between gap-2">
                <ModalityBadge modality={modality} tone={isEnrolled ? 'default' : 'muted'} />
                {loading ? <span className="h-5 w-20 animate-pulse rounded-full bg-raised" aria-hidden /> : <EnrollmentStatusPill status={status} />}
              </div>
              <div>
                <p className="text-card-title text-foreground">{MODALITY_LABEL[modality]}</p>
                <p className="text-meta text-muted-foreground">{FACTOR_DETAIL[modality]}</p>
              </div>
              {!loading && !isEnrolled && (
                <Link to={`/building/${building.id}/register?focus=${modality}`} className="mt-auto text-xs font-medium text-primary hover:underline">
                  Enroll {MODALITY_LABEL[modality].toLowerCase()}
                </Link>
              )}
            </motion.li>
          )
        })}
      </motion.ul>

      {/* How a decision is made */}
      <motion.div
        variants={rise}
        initial="hidden"
        animate="show"
        className="mx-auto mb-8 flex max-w-3xl flex-wrap items-center justify-center gap-x-2 gap-y-1 text-xs text-muted-foreground"
        aria-label="Face, voice and hand gesture are fused into one access decision"
      >
        <span className="flex items-center gap-1.5">
          <ScanFace className="h-3.5 w-3.5" strokeWidth={1.5} aria-hidden />
          <Mic className="h-3.5 w-3.5" strokeWidth={1.5} aria-hidden />
          <Hand className="h-3.5 w-3.5" strokeWidth={1.5} aria-hidden />
          Presented factors
        </span>
        <ArrowRight className="h-3.5 w-3.5" strokeWidth={1.5} aria-hidden />
        <span className="flex items-center gap-1.5">
          <Layers className="h-3.5 w-3.5" strokeWidth={1.5} aria-hidden /> Multimodal fusion
        </span>
        <ArrowRight className="h-3.5 w-3.5" strokeWidth={1.5} aria-hidden />
        <span className="flex items-center gap-1.5">
          <ShieldCheck className="h-3.5 w-3.5" strokeWidth={1.5} aria-hidden /> Access decision
        </span>
      </motion.div>

      {error && (
        <p role="alert" className="mx-auto mb-6 flex max-w-3xl items-center gap-2 rounded-lg border border-danger/30 bg-danger/10 p-3 text-sm text-danger">
          <AlertCircle className="h-4 w-4 shrink-0" strokeWidth={1.5} aria-hidden /> {error}
        </p>
      )}
      {!loading && !error && !canAuthenticate && (
        <p className="mx-auto mb-6 flex max-w-3xl items-start gap-2 rounded-lg border border-warning/30 bg-warning/10 p-3 text-sm text-warning">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" strokeWidth={1.5} aria-hidden /> Enroll at least one biometric factor before authenticating.
        </p>
      )}

      <motion.div variants={rise} initial="hidden" animate="show" className="flex flex-col items-center gap-3">
        {canAuthenticate ? (
          <Link to={`/building/${building.id}/authenticate`} className="btn btn-primary min-w-60 px-8">
            Begin Authentication
            <ArrowRight className="h-4 w-4" strokeWidth={1.75} aria-hidden />
          </Link>
        ) : (
          <Link to={`/building/${building.id}/register`} className="btn btn-primary min-w-60 px-8">
            Enroll Biometrics
            <ArrowRight className="h-4 w-4" strokeWidth={1.75} aria-hidden />
          </Link>
        )}
        <div className="flex flex-wrap items-center justify-center gap-x-5 gap-y-1">
          {canAuthenticate && (
            <Link to={`/building/${building.id}/register`} className="inline-flex min-h-11 items-center text-xs text-muted-foreground transition-colors hover:text-foreground">
              Manage biometric enrollment
            </Link>
          )}
          {/* Always available, even with only one known user - a different person enrolling never
              overwrites this user's templates: the registration page first asks for the new person's
              name and creates a new backend user (with its own internal id) for them. */}
          <Link to={`/building/${building.id}/register?new=1`} className="inline-flex min-h-11 items-center text-xs font-medium text-primary hover:underline">
            + New Registration
          </Link>
        </div>
      </motion.div>
    </div>
  )
}
