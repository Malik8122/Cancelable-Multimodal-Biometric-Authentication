# Backend API

FastAPI REST API over `template_protection/` and `embeddings/pipelines.py`.
Covers both modalities (face, voice); multimodal
fusion (`POST /authenticate/fusion`) and the Phase 3A dashboard are documented
once built (`docs/ROADMAP.md`). Run locally with:

```bash
cp .env.example .env   # then set a real MASTER_SECRET
uvicorn backend.main:app --reload
```

Interactive docs (Swagger UI) are available at `http://127.0.0.1:8000/docs`
once running, auto-generated from the same Pydantic models documented below -
disabled entirely (along with `/redoc` and `/openapi.json`) when `ENV=production`,
see `backend/main.py::_is_production`.

**CORS**: only one explicitly allowed origin may call this API from a
browser - `CORS_ORIGIN` (a single origin, the production/deployment-facing
name: set it to your deployed frontend's exact URL) or `CORS_ALLOWED_ORIGINS`
(comma-separated, for local/multi-origin development; defaults to
`http://localhost:5173`, the Vite dev server) - see
`backend/main.py::_resolve_cors_origins`. Never `"*"`.

See [`README.md`](../README.md#deployment-render--vercel) for the full
Render (backend) + Vercel (frontend) deployment guide.

## Scope note: "auth" here means biometric verification

`backend/auth/` exists because the spec calls for an `auth/` folder, but
there is no user login, session, or token system anywhere in this project.
Every "authenticate" in this API means *does this biometric sample match
what's enrolled*, not *is this HTTP caller who they claim to be*. If this
project ever adds real API-caller authentication (e.g. an API key per
integrating application), that would be a separate, additive concern, not a
change to what's described here.

## Conventions

- All biometric-accepting endpoints take `multipart/form-data` with an
  `image` file field plus ordinary form fields - not JSON with a base64
  payload. For voice, `image` is still the field name (for route-handler
  symmetry with face) but its *content* is a WAV audio file, not
  a picture - see `backend/utils.py::decode_biometric_sample`.
- `application_id` is optional on every request; omitting it falls back to
  `backend/config.py`'s `Settings.application_id` (`"capstone-demo"` by
  default).
- `modality` is one of `"face"`, `"voice"`.
- Only protected templates are ever persisted or returned - no endpoint
  response includes a raw image, a raw embedding, or template bytes.

## `POST /enroll`

Preprocess -> embed **once** -> write that modality's template into each **template
set** (see `docs/MULTI_TEMPLATE_ARCHITECTURE.md`). A new user gets `TEMPLATE_POOL_SIZE`
(default 4) sets: Set 1 `ACTIVE`, the rest `STANDBY`. Enroll each modality in turn; every
live set then holds a template for all of them. A modality enrolled later joins every live
set; re-enrolling one replaces its templates inside them (key versions are never reused).

**Request** (multipart/form-data):

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | string | yes | External identifier, e.g. `"U001"` |
| `modality` | string | yes | `face` \| `voice` |
| `application_id` | string | no | Defaults to `Settings.application_id` |
| `image` | file | yes | PNG/JPEG/BMP for face, WAV for voice; under `Settings.max_upload_size_bytes` |

**Response** `200 OK` (never contains template bytes):

```json
{
  "success": true,
  "user_id": "U001",
  "modality": "face",
  "templates_created": 4,
  "active_template_set_version": 1,
  "standby_template_set_versions": [2, 3, 4],
  "template_version": 1,
  "key_version": 1,
  "template_id": "1294340e-9a34-4cb9-bbb2-00c9f6259154"
}
```

**Face: one-time five-pose enrollment.** Send `pose_front`, `pose_left`, `pose_right`, `pose_up`, `pose_down` (face only; `image` is then
not needed). Per pose the backend runs the existing MTCNN detection + alignment and one FaceNet embedding, and rejects **only** a capture with
no detectable face (`NO_FACE`) or a blurry aligned crop (`BLURRY`). The valid embeddings are averaged and L2-normalized into a **centroid**, the
temporary embeddings are discarded immediately, and the T1-T4 template sets are generated from the centroid alone through the unchanged
HKDF + BioHash path - only protected templates are stored. Response: `poses_valid` and `pose_results` (`[{"pose", "status"}]`). At least 3 valid
poses are needed; fewer -> `422 {"status": "FACE_CAPTURE_REJECTED", "pose_results": [...], "detail": "... Nothing was enrolled ..."}` and nothing
is stored. A single `image` still enrolls a face from one capture. Authentication is unchanged: one live image, one embedding, compared with the
ACTIVE template.

`POST /enroll/face/check-pose` (form: `pose`, `image`) returns `{"pose", "status": "VALID" | "NO_FACE" | "BLURRY", "valid", "detail"}` so the UI can
ask for an immediate retake. It stores nothing.

**Voice: two recordings, graded consistency check.** `confirm_image` (voice only) is the second recording. Before anything is
stored the backend compares the two by the **cosine similarity of their ECAPA-TDNN embeddings** - not waveforms, not mel
spectrograms, not BioHash. Speaker embeddings tolerate natural variation (pace, phrasing, mild noise), so only clearly
mismatched recordings are rejected:

| ECAPA embedding cosine of the two recordings | Band | Result |
|---|---|---|
| >= 0.85 | **Excellent** | enrolled |
| 0.75 - 0.84 | **Good** | enrolled |
| 0.60 - 0.74 | **Fair** | `409 LOW_QUALITY_WARNING` - nothing stored unless the user continues (`accept_low_quality=true`) or re-records |
| < 0.60 | **Poor** | `422 ENROLLMENT_INCONSISTENT` - rejected, nothing stored |

Templates are stored only for the first three outcomes (Excellent, Good, and Fair once the user continues), always from the first
recording and always as the usual T1 ACTIVE + T2-T4 STANDBY pool. A Fair or Poor result stores nothing, so the modality stays as it
was (a rejected *re-*enrollment keeps the previous enrollment). Responses:

- `200` -> `recording_quality`: `EXCELLENT` \| `GOOD` (or `FAIR` after Continue).
- `409` -> `{"status": "LOW_QUALITY_WARNING", "recording_quality": "FAIR", "detail": "Your recordings are usable, but quality is lower than recommended."}`.
  Repeat the same request with `accept_low_quality=true` to continue. A warning is not a failed attempt (status stays `NOT_REGISTERED`).
- `422` -> `{"status": "ENROLLMENT_INCONSISTENT", "recording_quality": "POOR", "enrollment_status": "RETRY_REQUIRED", "detail": "The two recordings appear to be from different speakers or are too noisy. Nothing was enrolled - please record again."}`;
  `accept_low_quality` cannot override it.

The raw cosine is returned only with `DEBUG_SCORES=true`; production exposes just the band. This check gates enrollment only - it changes no
authentication threshold, no fusion logic and no template generation.

`templates_created` is the number of sets this modality was written into.
`template_version` (BioHash format version) / `key_version` / `template_id` describe this
modality's template in the ACTIVE set.

## `POST /authenticate`

Preprocess -> embed -> transform under the **ACTIVE set's** key -> compare with the ACTIVE
set's template (standby and revoked sets are never read; templates of different sets are
never mixed).

**Request**: same shape as `/enroll`.

**Response** `200 OK` - the single public authentication response, shared by
`/authenticate`, `/verify/*` and `/authenticate/fusion`:

```json
{
  "user_id": "U001",
  "status": "ACCESS_GRANTED",
  "authenticated": true,
  "fusion_similarity": 0.93,
  "fusion_threshold": 0.9,
  "fusion_policy": "ALL_REQUIRED",
  "matched_modalities": ["face"],
  "modalities_used": ["face"],
  "template_set_version": 1,
  "key_version": 1,
  "authentication_time_ms": 412
}
```

`fusion_similarity` is the mean of the per-modality scores on the estimated-cosine scale
(higher = better; for one modality, that modality's score) - see
[BIOMETRIC_METRICS.md](BIOMETRIC_METRICS.md): face is decided on an estimated cosine similarity
(>= 0.80), voice on an estimated Euclidean distance (<= 0.75, converted to `1 - d^2/2` before
fusion), both derived from the 256-bit template Hamming comparison. `display_name` (the user's
human-readable name) is included **only** when access is granted. `template_set_version` is the ACTIVE set that matched; `key_version` the highest
HKDF key version among its templates. **Per-modality similarities, per-modality thresholds,
distances (`fusion_distance` included), `results` and `fused_score` are not returned** unless the
server runs with `DEBUG_SCORES=true`; they are always written to the audit log. If nothing is
enrolled this returns `200` with `"authenticated": false, "fusion_similarity": 0.0`.

## Biometric protocols (enrollment vs authentication)

Enrollment and authentication are separate protocols (frontend: `frontend/src/config/protocol.ts`).

| | Registration / enrollment | Authentication |
|---|---|---|
| **Step 1 - Face** | 5 guided samples, averaged into one centroid (`POST /enroll/face`) | 1 capture |
| **Step 2 - Voice** | 2 sentences - sample + consistency check (`POST /enroll`, `image` + `confirm_image`) | **1 sentence** (`voice_audio`) |
| **Step 3 - Dynamic hand gesture** (optional) | **3 independent Z gesture samples** - separate performances, each kept as its own sequence (`POST /enroll/hand`) | 1 Z gesture sample |

Entering the user's name comes first but is account setup, not a numbered biometric step. The per-modality results of
one authentication attempt are fused under the unchanged fusion policy (fail-closed: a capture error is never a match).

## `POST /authenticate/fusion`

**The user chooses the factors.** Buildings define no biometric policy; `building_id` is an optional label (recorded in the audit
log; unknown -> `404`). Submit any of `face_image`, `voice_audio`, `hand_gesture` (none -> `422`) - exactly ONE sample
per modality. Voice authentication is one sentence: `voice_audio_2` is refused with `422` ("Voice authentication uses
exactly one sentence") rather than silently ignored.

1. a submitted modality that is not enrolled -> **`409`** `{"status": "ENROLLMENT_REQUIRED", "authentication_state": ..., "detail": ...,
   "submitted_modalities": [...], "enrolled_modalities": [...], "missing_modalities": [...]}`. That modality is not authenticated,
   nothing is decoded or evaluated, and the attempt is audited as `ENROLLMENT_REQUIRED`. **This is not an authentication failure.**
2. otherwise exactly the submitted modalities are authenticated against the ACTIVE template set and fused: `authentication_state`
   is `ACCESS_GRANTED` (all verified) or `ACCESS_DENIED`, both `200`.

The same 409 rule applies to `/authenticate` and `/verify/*`.

Fields: `user_id`, optional `application_id`, `building_id`, `fusion_policy`
(`ALL_REQUIRED` default \| `WEIGHTED` \| `AT_LEAST_TWO`), and the captures. Same response as `/authenticate` (below).
`fusion_similarity = mean(s_m)`, `fusion_distance =
1 - fusion_similarity`, `fusion_threshold = mean(t_m)` over the submitted modalities; under `ALL_REQUIRED` each submitted
modality must individually pass its own threshold.

## `POST /verify/face` · `POST /verify/voice`

Identical to `/authenticate`, with the modality fixed by the URL (the request omits `modality`).
Same response shape. `/verify/voice`'s `image` field is a WAV file.

## Hand gesture: `POST /enroll/hand` · `POST /enroll/hand/check` · `POST /verify/hand`

Uploads are landmark JSON (`application/json`, format in `docs/HAND_GESTURE_MODALITY.md` §4), never video.

- `POST /enroll/hand`: `user_id`, optional `application_id`, `attempts` × 3 (one file per Z execution). If any execution
  fails the capture gate (including the Z-structure check), or one sample is unlike both others -> `422 {"status": "HAND_CAPTURE_REJECTED", "verdict": <failure code>, "attempt": n, "capture_error": bool, "detail"}`
  and nothing is stored; wrong count -> `422`. Success: the `/enroll` response plus a `hand_gesture` summary. `POST /enroll`
  with `modality=hand` is refused (`422`) and points here.
- `POST /enroll/hand/check`: `gesture`, `attempt` -> whether one execution passes the capture gate and Z check (no storage).
- `POST /verify/hand`: `user_id`, optional `application_id`, `gesture` (one execution). Same response as `/verify/*`, with a
  `hand_gesture` block (status, decision `MATCH` / `GESTURE_MISMATCH` / failure code, frames, duration, timings; the DTW
  distances and threshold only with `DEBUG_SCORES=true`). Capture failures (`NO_HAND_DETECTED`, `INSUFFICIENT_FRAMES`,
  `MALFORMED_CAPTURE`) -> `CAPTURE_ERROR`; `CAPTURE_QUALITY_FAILURE` / `INVALID_TRAJECTORY` -> `QUALITY_INSUFFICIENT`; neither
  counts as a mismatch.
- `/authenticate/fusion`, `/revoke-template` and `/templates/*` accept the optional `hand_gesture` field.
  `fusion_policy=AT_LEAST_TWO` needs at least two submitted modalities (`422` otherwise); at least two must pass.

## Template-set management (biometric authorization)

There is no login. Every mutating template-set call below must carry a **fresh capture of every
modality in the ACTIVE set** as multipart fields `face_image` and `voice_audio`. Each capture is authenticated against its ACTIVE-set template; a missing/extra
modality or any mismatch returns **`403`** (`"Biometric authorization failed."`) and changes
nothing. Attempts are audit-logged. Nothing here returns template bytes.

### `POST /revoke-template`

Fields: `user_id`, optional `application_id`, `reason`, plus the authorization captures.
Revokes the whole ACTIVE set and activates the oldest STANDBY set for **every modality at once**.

```json
{
  "success": true,
  "user_id": "U001",
  "revoked_template_set_version": 1,
  "new_active_template_set_version": 2,
  "remaining_standby_template_sets": 2
}
```

`403` authorization failed; `404` nothing enrolled; **`409` `"Template set pool exhausted.
Re-enrollment required."`** when no STANDBY set remains (nothing changes).

### `GET /templates/{user_id}`

Optional query `application_id`. Read-only (no authorization). The template set pool:

```json
{
  "user_id": "U001", "application_id": "capstone-demo", "pool_size": 4,
  "active_template_set_version": 2, "standby_count": 2,
  "sets": [
    {"template_set_version": 1, "status": "REVOKED", "modalities": ["face", "voice"],
     "key_versions": {"face": 1, "voice": 1}, "template_group_id": "...",
     "created_at": "...", "activated_at": "...", "revoked_at": "...", "revoked_reason": "revoked by user"},
    {"template_set_version": 2, "status": "ACTIVE", "...": "..."}
  ]
}
```

`404` for an unknown user or one with no sets.

### `POST /templates/{user_id}/activate/{version}`

Fields: optional `application_id` + the authorization captures. Promotes STANDBY set `version`
to ACTIVE; the previous ACTIVE set becomes REVOKED (`superseded by manual activation`).
`403` / `404` (not a STANDBY set). Returns `{success, user_id, previous_active_template_set_version,
new_active_template_set_version, remaining_standby_template_sets}`.

### `POST /templates/{user_id}/generate`

Fields: optional `application_id` + the authorization captures. Adds **one** complete new STANDBY
set (every modality of the ACTIVE set) built from those same captures, under fresh key versions.
`409` when the pool already holds `TEMPLATE_POOL_SIZE` live sets; `403` / `404` as above. Returns
`{success, user_id, new_template_set_version, standby_template_set_versions,
active_template_set_version}`. (`/templates/{user_id}/replenish` remains as a hidden alias.)
This is also how a migrated user with a pool of one set gets standby sets.

## Users and display names

A display name is only a label; the internal `user_id` stays the identifier (primary key, key-derivation input, what
templates attach to), so two users may share a name. Stored in `users.username`; users registered before names existed
have none and are shown as `User <last 6 id characters>`. Validation (`backend/display_names.py`): trimmed, inner
whitespace collapsed to one space, 1-64 characters, letters (any script), spaces and `' ’ - .`, starting with a letter.

### `POST /users`

Starts a registration. Body `{"display_name": "Sanya Malik"}` -> `201`
`{"user_id": "USER-3F9A1C0B7D2E", "display_name": "Sanya Malik"}` (the id is server-generated). Invalid name -> `422`.
The face / voice samples are then enrolled under the returned `user_id` with `/enroll/face` and `/enroll`.

### `POST /user/{user_id}/display-name`

Names or renames an existing user (e.g. one registered before names existed). Same body and response; `404` for an
unknown user, `422` for an invalid name. Templates are not touched.

## Enrollment profile and buildings

### `GET /user/{user_id}/enrollment-status`

Optional query `application_id`. Which modalities the user has enrolled and the status of each (always `200`; an unknown user
simply has nothing registered):

```json
{"user_id": "USER001", "application_id": "capstone-demo",
 "modalities": {"face": true, "voice": false},
 "statuses": {"face": "REGISTERED", "voice": "RETRY_REQUIRED"},
 "display_name": "Sanya Malik", "has_display_name": true}
```

`display_name` is the fallback `User <short id>` when `has_display_name` is `false`.

`statuses`: `NOT_REGISTERED`, `REGISTERED`, `UPDATED` (re-enrolled), `RETRY_REQUIRED` (voice only - the last enrollment failed the
two-recording check; nothing was stored).

### `GET /buildings` and `GET /building/{building_id}`

Buildings are authentication context only - no `required_modalities`:

```json
[{"id": "national_data_center", "name": "National Data Centre", "description": "...", "clearance_level": "IV"}, "..."]
```

`404` for an unknown building. The source is `config/buildings.json`; a file that defines `required_modalities` is rejected.

### Audit log entries

`GET /audit/{user_id}` and `/audit/system` entries add `authentication_state` (`ACCESS_GRANTED` / `ACCESS_DENIED` /
`ENROLLMENT_REQUIRED`), `submitted_modalities`, `enrolled_modalities`, `authenticated_modalities`, `template_set_version`,
`template_set_status` (alongside `building_id`, `fusion_similarity`, `latency_ms`). Per-modality similarities are returned only
with `DEBUG_SCORES=true`.

## `GET /user/{id}`

**Response** `200 OK`:

```json
{
  "user_id": "U001",
  "enrolled_modalities": [
    {
      "modality": "face",
      "application_id": "capstone-demo",
      "template_version": 1,
      "key_version": 1,
      "created_at": "2026-01-01T00:00:00Z"
    }
  ]
}
```

`404 Not Found` if `user_id` has never been enrolled in anything.

## `DELETE /user/{id}`

Deletes the user and every protected template they own (cascading delete -
see `backend/database/models.py::User.templates`).

**Response** `200 OK`:

```json
{ "success": true, "user_id": "U001", "templates_deleted": 2 }
```

Returns `templates_deleted: 0` (still `200 OK`, not `404`) if `user_id`
doesn't exist - deleting something that's already gone is treated as
already-satisfied, not an error.

## `GET /health`

Liveness check: `{"status": "ok"}`. No auth, no dependencies touched.

## Error responses

| Status | When |
|---|---|
| `400 Bad Request` | Uploaded file isn't a decodable image |
| `404 Not Found` | `GET /user/{id}` for an unenrolled user |
| `409 Conflict` | Two concurrent template writes raced for the same (user, modality, application) — retry; or the template set pool is exhausted / already full |
| `413 Request Entity Too Large` | Upload exceeds `Settings.max_upload_size_bytes` |
| `415 Unsupported Media Type` | Upload's content-type isn't in `Settings.allowed_content_types` (images) or `Settings.allowed_audio_content_types` (voice) |
| `422 Unprocessable Entity` | Unknown `modality` value, or a missing required field |

Every error body is `{"detail": "<message>"}` (FastAPI's default).

## Configuration

See `.env.example` for the full list of environment variables
(`MASTER_SECRET`, `DATABASE_URL`, `APPLICATION_ID`,
`FACE_MODEL_PATH`/`VOICE_MODEL_PATH`,
`TEMPLATE_BITS`, `MATCH_THRESHOLD`, `FACE_COSINE_THRESHOLD` (default `0.80`, estimated cosine,
higher = better), `VOICE_EUCLIDEAN_THRESHOLD` (default `0.75`, estimated Euclidean distance,
lower = better)) and `backend/config.py` for what each one
does and its default. `MASTER_SECRET` has no default and must be set - the
server fails to start without it, by design.

Multi-template settings (environment variables): `TEMPLATE_POOL_SIZE` (default `4`,
template sets created at enrollment) and `DEBUG_SCORES` (default
`false`; when `true`, responses and the audit API also carry the internal
per-modality scores - never enable in production).

Building policy file: `BUILDINGS_CONFIG_PATH` (default `config/buildings.json`).
