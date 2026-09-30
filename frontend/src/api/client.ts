// The only place in this app that calls `fetch`. Every biometric endpoint
// call goes through here so there is exactly one spot that knows the base
// URL, and so no component can accidentally hand-roll its own request.
//
// This layer does zero score/decision computation - it only shapes requests
// and parses responses. Fusion, FAR/FRR/EER, and match/no-match decisions
// are all computed server-side (see backend/api/fusion.py,
// backend/services/base_service.py) and simply passed through as-is.

import type {
  ActivateResponse,
  AuditHistoryResponse,
  AuthenticateResponse,
  AuthenticationOutcome,
  BuildingInfo,
  EnrollResponse,
  EnrollmentRequiredResponse,
  FacePose,
  FacePoseCheckResponse,
  HandGestureCheckResponse,
  VoiceQualityCheckResponse,
  EnrollmentStatusResponse,
  FusionAuthenticateResponse,
  FusionPolicy,
  GenerateSetResponse,
  Modality,
  ModalityMetricsResponse,
  RevokeResponse,
  SystemAuditResponse,
  SystemHealthResponse,
  TemplateSetPoolResponse,
  UserModalitiesResponse,
  UserProfileResponse,
} from './types'
import { ApiError } from './types'

export const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, init)
  if (!response.ok) {
    let detail = response.statusText
    let body: unknown
    try {
      body = await response.json()
      const parsed = (body as { detail?: unknown }).detail
      if (typeof parsed === 'string') detail = parsed
    } catch {
      // Body wasn't JSON (e.g. a proxy error page) - keep statusText.
    }
    throw new ApiError(response.status, detail, body)
  }
  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

function biometricForm(
  userId: string,
  applicationId: string,
  sample: Blob,
  filename: string,
  extra?: Record<string, string>,
): FormData {
  const form = new FormData()
  form.append('user_id', userId)
  form.append('application_id', applicationId)
  if (extra) {
    for (const [key, value] of Object.entries(extra)) form.append(key, value)
  }
  form.append('image', sample, filename)
  return form
}

export interface EnrollOptions {
  /** Voice: the second recording. Enrollment is all-or-nothing - inconsistent recordings store nothing (422). */
  confirm?: { sample: Blob; filename: string }
  /** Voice: continue with FAIR-quality recordings (the backend otherwise answers 409 LOW_QUALITY_WARNING, storing nothing). */
  acceptLowQuality?: boolean
}

export async function enroll(
  modality: Modality,
  userId: string,
  applicationId: string,
  sample: Blob,
  filename: string,
  options?: EnrollOptions,
): Promise<EnrollResponse> {
  const form = biometricForm(userId, applicationId, sample, filename, { modality })
  if (options?.confirm) form.append('confirm_image', options.confirm.sample, options.confirm.filename)
  if (options?.acceptLowQuality) form.append('accept_low_quality', 'true')
  return request<EnrollResponse>('/enroll', { method: 'POST', body: form })
}

export async function verify(
  modality: Modality,
  userId: string,
  applicationId: string,
  sample: Blob,
  filename: string,
): Promise<AuthenticateResponse> {
  const form = biometricForm(userId, applicationId, sample, filename)
  return request<AuthenticateResponse>(`/verify/${modality}`, { method: 'POST', body: form })
}

// One-time face enrollment from the five guided poses. Send them in one request; the backend embeds each valid pose, averages
// the embeddings into a centroid, discards them, and generates the template sets from the centroid alone.
export async function enrollFace(userId: string, applicationId: string, poses: Record<FacePose, Blob>): Promise<EnrollResponse> {
  const form = new FormData()
  form.append('user_id', userId)
  form.append('application_id', applicationId)
  form.append('modality', 'face')
  for (const [pose, blob] of Object.entries(poses) as [FacePose, Blob][]) form.append(`pose_${pose}`, blob, `face-${pose}.jpg`)
  return request<EnrollResponse>('/enroll', { method: 'POST', body: form })
}

// Immediate verdict for ONE captured pose (VALID / NO_FACE / BLURRY) so the guided UI can ask for a retake. Stores nothing.
export async function checkFacePose(pose: FacePose, sample: Blob, filename: string): Promise<FacePoseCheckResponse> {
  const form = new FormData()
  form.append('pose', pose)
  form.append('image', sample, filename)
  return request<FacePoseCheckResponse>('/enroll/face/check-pose', { method: 'POST', body: form })
}

// Capture-quality verdict for ONE voice recording, so the UI can ask for a retake of just that sentence. Stores nothing;
// the backend re-checks every recording when it is submitted.
export async function checkVoiceQuality(sample: Blob, filename: string, sentence?: number): Promise<VoiceQualityCheckResponse> {
  const form = new FormData()
  form.append('audio', sample, filename)
  if (sentence) form.append('sentence', String(sentence))
  return request<VoiceQualityCheckResponse>('/voice/check-quality', { method: 'POST', body: form })
}

/** One captured sample of one presented factor. One per modality, except voice: up to two sentences. */
export interface FusionSample {
  modality: Modality
  sample: Blob
  filename: string
}

const FUSION_FIELD_NAME: Record<Modality, string> = {
  face: 'face_image',
  voice: 'voice_audio',
  hand: 'hand_gesture',
}

// --- Dynamic hand gesture (landmark sequences from MediaPipe in the browser; never video) -------------------
// Three independent Z samples in one request; the backend validates each, keeps all three as separate sequences
// under each template set's key, and answers 422 HAND_CAPTURE_REJECTED (with the sample number) when one is unusable
// or unlike both others, or 422 when the same captured sequence appears twice - nothing is stored then.
export async function enrollHand(userId: string, applicationId: string, attempts: Blob[]): Promise<EnrollResponse> {
  const form = new FormData()
  form.append('user_id', userId)
  form.append('application_id', applicationId)
  attempts.forEach((blob, i) => form.append('attempts', blob, `gesture-${i + 1}.json`))
  return request<EnrollResponse>('/enroll/hand', { method: 'POST', body: form })
}

// The capture gate's verdict for ONE gesture sample so the UI can ask for a retake immediately. Stores nothing.
export async function checkHandGesture(gesture: Blob, attempt?: number): Promise<HandGestureCheckResponse> {
  const form = new FormData()
  form.append('gesture', gesture, 'gesture.json')
  if (attempt) form.append('attempt', String(attempt))
  return request<HandGestureCheckResponse>('/enroll/hand/check', { method: 'POST', body: form })
}

// One Z gesture sample, compared (DTW) with the three enrolled ones. Same response as every /verify/* route.
export async function verifyHand(userId: string, applicationId: string, gesture: Blob): Promise<AuthenticateResponse> {
  const form = new FormData()
  form.append('user_id', userId)
  form.append('application_id', applicationId)
  form.append('gesture', gesture, 'gesture.json')
  return request<AuthenticateResponse>('/verify/hand', { method: 'POST', body: form })
}

export async function authenticateFusion(
  userId: string,
  applicationId: string,
  samples: FusionSample[],
  options?: { fusionPolicy?: FusionPolicy; buildingId?: string; signal?: AbortSignal },
): Promise<AuthenticationOutcome> {
  const form = new FormData()
  form.append('user_id', userId)
  form.append('application_id', applicationId)
  if (options?.fusionPolicy) form.append('fusion_policy', options.fusionPolicy)
  if (options?.buildingId) form.append('building_id', options.buildingId)
  // Authentication protocol: one sample per presented modality, except voice - up to two sentences (voice_audio,
  // voice_audio_2), which the backend combines into one embedding.
  const counts: Partial<Record<Modality, number>> = {}
  for (const { modality, sample, filename } of samples) {
    const n = (counts[modality] ?? 0) + 1
    if (n > (modality === 'voice' ? 2 : 1)) {
      throw new Error(`Authentication takes ${modality === 'voice' ? 'at most two voice sentences' : `one ${modality} sample`}.`)
    }
    counts[modality] = n
    form.append(n === 2 ? 'voice_audio_2' : FUSION_FIELD_NAME[modality], sample, filename)
  }
  // `signal` is optional and passed straight through to fetch() (via
  // request()'s existing RequestInit parameter) - only the caller that wants
  // a client-side abort/timeout (AuthenticatePage.tsx) needs to pass one;
  // every other caller/endpoint is unaffected.
  try {
    return await request<FusionAuthenticateResponse>('/authenticate/fusion', { method: 'POST', body: form, signal: options?.signal })
  } catch (error) {
    // 409 ENROLLMENT_REQUIRED is a legitimate third outcome, not an error: hand it to the caller as a result.
    const body = error instanceof ApiError ? (error.body as Partial<EnrollmentRequiredResponse> | undefined) : undefined
    if (error instanceof ApiError && error.status === 409 && body?.status === 'ENROLLMENT_REQUIRED') {
      return body as EnrollmentRequiredResponse
    }
    throw error
  }
}

// --- Buildings (context) + the user's enrollment profile -----------------
export async function getBuildings(): Promise<BuildingInfo[]> {
  return request<BuildingInfo[]>('/buildings')
}

export async function getEnrollmentStatus(userId: string, applicationId: string): Promise<EnrollmentStatusResponse> {
  return request<EnrollmentStatusResponse>(
    `/user/${encodeURIComponent(userId)}/enrollment-status?application_id=${encodeURIComponent(applicationId)}`,
  )
}

// --- Users: internal id + human-readable display name -------------------
// Registration starts here: the backend generates the internal user_id; the name is only a label.
export async function createUser(displayName: string): Promise<UserProfileResponse> {
  return request<UserProfileResponse>('/users', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ display_name: displayName }),
  })
}

// Name an existing user (e.g. one registered before names existed). Templates are untouched.
export async function setDisplayName(userId: string, displayName: string): Promise<UserProfileResponse> {
  return request<UserProfileResponse>(`/user/${encodeURIComponent(userId)}/display-name`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ display_name: displayName }),
  })
}

// --- Template sets ---------------------------------------------------------
// Revoking, activating and generating template sets all require biometric
// authorization: a fresh capture of EVERY modality the ACTIVE set contains
// (the backend answers 403 otherwise). There is no login layer.
export type AuthorizationCaptures = Partial<Record<Modality, { blob: Blob; filename: string }>>

function authorizationForm(userId: string, applicationId: string, captures: AuthorizationCaptures): FormData {
  const form = new FormData()
  form.append('user_id', userId)
  form.append('application_id', applicationId)
  for (const [modality, capture] of Object.entries(captures) as [Modality, { blob: Blob; filename: string }][]) {
    form.append(FUSION_FIELD_NAME[modality], capture.blob, capture.filename)
  }
  return form
}

export async function getTemplateSets(userId: string, applicationId: string): Promise<TemplateSetPoolResponse | null> {
  try {
    return await request<TemplateSetPoolResponse>(
      `/templates/${encodeURIComponent(userId)}?application_id=${encodeURIComponent(applicationId)}`,
    )
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null
    throw error
  }
}

// Revoke the ACTIVE set; the oldest STANDBY set becomes ACTIVE for every modality (409 when none is left).
export async function revokeTemplateSet(
  userId: string,
  applicationId: string,
  captures: AuthorizationCaptures,
): Promise<RevokeResponse> {
  return request<RevokeResponse>('/revoke-template', { method: 'POST', body: authorizationForm(userId, applicationId, captures) })
}

export async function activateTemplateSet(
  userId: string,
  applicationId: string,
  version: number,
  captures: AuthorizationCaptures,
): Promise<ActivateResponse> {
  return request<ActivateResponse>(`/templates/${encodeURIComponent(userId)}/activate/${version}`, {
    method: 'POST',
    body: authorizationForm(userId, applicationId, captures),
  })
}

export async function generateTemplateSet(
  userId: string,
  applicationId: string,
  captures: AuthorizationCaptures,
): Promise<GenerateSetResponse> {
  return request<GenerateSetResponse>(`/templates/${encodeURIComponent(userId)}/generate`, {
    method: 'POST',
    body: authorizationForm(userId, applicationId, captures),
  })
}

export async function getUser(userId: string): Promise<UserModalitiesResponse | null> {
  try {
    return await request<UserModalitiesResponse>(`/user/${encodeURIComponent(userId)}`)
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null
    throw error
  }
}

export async function getMetrics(modality: Modality): Promise<ModalityMetricsResponse> {
  return request<ModalityMetricsResponse>(`/metrics/${modality}`)
}

export async function getSystemHealth(): Promise<SystemHealthResponse | null> {
  try {
    return await request<SystemHealthResponse>('/system/health')
  } catch {
    return null
  }
}

export async function getUserAuditHistory(userId: string, limit = 20): Promise<AuditHistoryResponse | null> {
  try {
    return await request<AuditHistoryResponse>(`/audit/${encodeURIComponent(userId)}?limit=${limit}`)
  } catch (error) {
    if (error instanceof ApiError) return null
    throw error
  }
}

export async function getSystemAuditHistory(limit = 50): Promise<SystemAuditResponse | null> {
  try {
    return await request<SystemAuditResponse>(`/audit/system?limit=${limit}`)
  } catch (error) {
    if (error instanceof ApiError) return null
    throw error
  }
}

export async function deleteUserAuditHistory(userId: string): Promise<void> {
  await request<void>(`/audit/${encodeURIComponent(userId)}`, { method: 'DELETE' })
}
