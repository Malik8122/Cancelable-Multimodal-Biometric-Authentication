import { AlertCircle, Loader2, Plus, ShieldOff, X } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  activateTemplateSet,
  generateTemplateSet,
  getTemplateSets,
  revokeTemplateSet,
  type AuthorizationCaptures,
} from '../api/client'
import { ApiError, type Modality, type TemplateSetInfo, type TemplateSetPoolResponse } from '../api/types'
import { StatusBadge, TemplateSetCard } from '../components/biometric/TemplateSetCard'
import { FACTORS, MODALITY_LABEL } from '../config/buildings'
import { FaceCapture } from '../components/capture/FaceCapture'
import { VoiceCapture } from '../components/capture/VoiceCapture'
import { HandGestureCapture } from '../components/capture/HandGestureCapture'
import { payloadBlob } from '../hand/landmarker'
import { useSession } from '../context/SessionContext'
import { PageHeader } from '../components/layout/PageHeader'

const APPLICATION_ID = 'ncisn-security-network'

type PendingAction = { kind: 'revoke' } | { kind: 'generate' } | { kind: 'activate'; version: number }

const ACTION_LABEL: Record<PendingAction['kind'], string> = {
  revoke: 'Revoke Active Set',
  generate: 'Generate New Template Set',
  activate: 'Activate Set',
}

interface Notice {
  kind: 'ok' | 'error'
  text: string
}

// The template pool at a glance: one row per biometric, one column per template (T1..TN). Each cell is that
// modality's template in that set and its status - T1 ACTIVE, T2-T4 STANDBY, revoked ones REVOKED. A modality that is
// not enrolled has no cell (-). Status only; template values are never sent to the browser.
function TemplateMatrix({ sets }: { sets: TemplateSetInfo[] }) {
  const modalities = FACTORS.filter((m) => sets.some((s) => s.modalities.includes(m)))
  return (
    <div className="surface-card mb-6 overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-xs text-muted-foreground">
            <th className="px-4 py-2.5 font-medium">Biometric</th>
            {sets.map((s) => (
              <th key={s.template_set_version} className="px-4 py-2.5 font-medium">
                T{s.template_set_version}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {modalities.map((m) => (
            <tr key={m} className="border-b border-border/50 last:border-0">
              <td className="px-4 py-2.5 text-foreground">{MODALITY_LABEL[m]}</td>
              {sets.map((s) => (
                <td key={s.template_set_version} className="px-4 py-2.5">
                  {s.modalities.includes(m) ? <StatusBadge status={s.status} /> : <span className="text-muted-foreground/50">-</span>}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function TemplateManagementPage() {
  const { userId } = useSession()
  const [pool, setPool] = useState<TemplateSetPoolResponse | null | undefined>(undefined)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [pending, setPending] = useState<PendingAction | null>(null)
  const [captures, setCaptures] = useState<AuthorizationCaptures>({})
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<Notice | null>(null)

  const refresh = useCallback(async () => {
    try {
      setPool(await getTemplateSets(userId, APPLICATION_ID))
      setLoadError(null)
    } catch (error) {
      setLoadError(error instanceof ApiError ? error.detail : 'The backend is unreachable.')
    }
  }, [userId])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const activeSet = pool?.sets.find((s) => s.status === 'ACTIVE')
  const liveSets = pool?.sets.filter((s) => s.status !== 'REVOKED').length ?? 0
  // Authorization = a fresh capture of every modality in the ACTIVE set.
  const authorizationModalities: Modality[] = activeSet?.modalities ?? []
  const allCaptured = authorizationModalities.length > 0 && authorizationModalities.every((m) => captures[m])

  const start = (action: PendingAction) => {
    setNotice(null)
    setCaptures({})
    setPending(action)
  }
  const cancel = () => {
    setPending(null)
    setCaptures({})
  }

  const confirm = async () => {
    if (!pending) return
    setBusy(true)
    setNotice(null)
    try {
      let text: string
      if (pending.kind === 'revoke') {
        const r = await revokeTemplateSet(userId, APPLICATION_ID, captures)
        text = `Template Set ${r.revoked_template_set_version} revoked. Template Set ${r.new_active_template_set_version} is now active for every modality (${r.remaining_standby_template_sets} standby left).`
      } else if (pending.kind === 'activate') {
        const r = await activateTemplateSet(userId, APPLICATION_ID, pending.version, captures)
        text = `Template Set ${r.new_active_template_set_version} is now active; the previous active set was revoked.`
      } else {
        const r = await generateTemplateSet(userId, APPLICATION_ID, captures)
        text = `Template Set ${r.new_template_set_version} generated and placed on standby.`
      }
      setNotice({ kind: 'ok', text })
      cancel()
      await refresh()
    } catch (error) {
      setNotice({
        kind: 'error',
        text: error instanceof ApiError ? error.detail : 'The backend is unreachable.',
      })
    } finally {
      setBusy(false)
    }
  }

  const capture = (modality: Modality) => (blob: Blob, filename: string) =>
    setCaptures((prev) => ({ ...prev, [modality]: { blob, filename } }))

  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6 sm:py-12">
      <PageHeader
        eyebrow="Cancelable credentials"
        title="Template Sets"
        description={
          <>
            Each template set is one complete multimodal credential. Only the active set authenticates; revoking it activates
            the next standby set for every biometric at once. Changes need a fresh capture of your biometrics to authorize.
            Operator: <span className="font-mono text-foreground">{userId}</span>
          </>
        }
      />

      {notice && (
        <div
          className={`mb-6 flex items-start justify-between gap-3 rounded-lg border p-3 text-sm ${
            notice.kind === 'ok' ? 'border-success/30 bg-success/10 text-success' : 'border-danger/30 bg-danger/10 text-danger'
          }`}
        >
          <span className="flex items-start gap-2">
            {notice.kind === 'error' && <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" strokeWidth={1.5} />}
            {notice.text}
          </span>
          <button onClick={() => setNotice(null)} aria-label="Dismiss">
            <X className="h-4 w-4" strokeWidth={1.5} />
          </button>
        </div>
      )}

      {loadError && <p className="text-center text-sm text-danger">{loadError}</p>}
      {pool === undefined && !loadError && (
        <p className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} /> Loading template sets&hellip;
        </p>
      )}
      {pool === null && (
        <div className="surface-card p-8 text-center">
          <p className="text-sm text-muted-foreground">No template sets are registered for this operator yet.</p>
          <Link to="/" className="mt-3 inline-block text-sm text-primary hover:underline">
            Choose a facility to register
          </Link>
        </div>
      )}

      {pool && (
        <>
          <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-muted-foreground">
              {activeSet ? `Set ${activeSet.template_set_version} active` : 'No active set'} &middot; {pool.standby_count} standby
              &middot; pool size {pool.pool_size}
            </p>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => start({ kind: 'revoke' })}
                disabled={busy || !activeSet || pool.standby_count === 0}
                type="button"
                className="btn btn-danger min-h-10 text-xs"
              >
                <ShieldOff className="h-3.5 w-3.5" strokeWidth={1.5} />
                Revoke Active Set
              </button>
              <button
                onClick={() => start({ kind: 'generate' })}
                disabled={busy || !activeSet || liveSets >= pool.pool_size}
                type="button"
                className="btn btn-secondary min-h-10 text-xs"
              >
                <Plus className="h-3.5 w-3.5" strokeWidth={1.5} />
                Generate New Template Set
              </button>
            </div>
          </div>

          {pool.standby_count === 0 && activeSet && (
            <p className="mb-4 text-xs text-muted-foreground">
              No standby sets left. Generate a new template set (authorized with a fresh capture) before revoking again.
            </p>
          )}

          <TemplateMatrix sets={pool.sets} />

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {pool.sets.map((s) => (
              <TemplateSetCard
                key={s.template_set_version}
                busy={busy}
                onActivate={() => start({ kind: 'activate', version: s.template_set_version })}
                set={{
                  version: s.template_set_version,
                  status: s.status,
                  modalities: s.modalities,
                  created: s.created_at,
                  activated: s.activated_at,
                  revoked: s.revoked_at,
                }}
              />
            ))}
          </div>

          {pending && (
            <section className="surface-card mt-8 border-primary/40 p-4 sm:p-5">
              <div className="mb-4 flex items-start justify-between gap-3">
                <div>
                  <h2 className="text-base font-medium text-foreground">
                    {ACTION_LABEL[pending.kind]}
                    {pending.kind === 'activate' ? ` ${pending.version}` : ''} - authorization
                  </h2>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Capture every biometric in the active template set. They must all match, otherwise the change is refused.
                  </p>
                </div>
                <button onClick={cancel} aria-label="Cancel" disabled={busy}>
                  <X className="h-4 w-4 text-muted-foreground" strokeWidth={1.5} />
                </button>
              </div>

              <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-2">
                {authorizationModalities.includes('face') && <FaceCapture mode="verify" onCapture={capture('face')} />}
                {authorizationModalities.includes('voice') && <VoiceCapture mode="verify" onCapture={capture('voice')} />}
                {authorizationModalities.includes('hand') && (
                  <HandGestureCapture mode="verify" onCapture={(payload) => capture('hand')(payloadBlob(payload), 'gesture.json')} />
                )}
              </div>

              <button
                onClick={() => void confirm()}
                disabled={!allCaptured || busy}
                type="button"
                className="btn btn-primary w-full"
              >
                {busy && <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} />}
                {busy ? 'Authorizing...' : `Authorize & ${ACTION_LABEL[pending.kind]}`}
              </button>
            </section>
          )}
        </>
      )}
    </div>
  )
}
