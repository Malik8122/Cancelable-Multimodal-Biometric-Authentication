# Template-Set Cancelable Authentication Architecture

Final architecture of the system (V3: user-driven flexible multimodal authentication). A user's credential is a
**pool of template sets**. Each *template set* is one complete multimodal
credential - one protected template for every enrolled modality (face,
voice) - and the set, not the individual modality, is the unit of
activation and revocation. Authentication returns **one decision and one fused
similarity**.

Unchanged: the face / voice models and checkpoints, the HKDF
implementation, the BioHash algorithm, the 256-bit template size, `ALL_REQUIRED`
as the default fusion policy, audit logging and runtime security validation, the
Render + Vercel deployment.

## 1. Vocabulary

| Term | Meaning |
|---|---|
| **Template set** | One multimodal credential: one protected template per enrolled modality. |
| **Template set version** | Its number, 1..N, unique per (user, application), never reused. This is the "template version" the API and UI show. |
| **Active template set** | The single set (per user + application) authentication uses. Face and voice templates are ACTIVE together. |
| **Standby template sets** | Valid, waiting. Never compared during authentication. Promoted oldest-first. |
| **Revoked template set** | Retired. Terminal - never becomes ACTIVE or STANDBY again. |
| **Key version** | The HKDF key version one template was generated under (per modality). Set *v* uses key *v* for a freshly enrolled user; key versions are never reused. |

## 2. Flexible multimodal authentication (V3)

Enrollment and authentication are **user-driven**. Three concepts, kept apart:

| Concept | What it is | Where it lives |
|---|---|---|
| **User Enrollment Profile** | Which of face / voice the *user* has enrolled, and the status of each. | Derived from the template rows (a modality is enrolled iff the user has a template for it in the ACTIVE set), plus `enrollment_events` for RETRY_REQUIRED. `backend/services/enrollment.py`, `GET /user/{id}/enrollment-status`. |
| **Building** | Authentication *context only*: `id`, `name`, `clearance_level`, `description`. **No biometric policy** - a building never requires or forbids a modality. | `config/buildings.json` -> `backend/buildings.py` -> `GET /buildings`, `GET /building/{id}`. The loader rejects any `required_modalities` key. |
| **Authentication Engine** | Authenticates and fuses **exactly the modalities the user submits** (all of which must be enrolled). One engine for every combination. | `POST /authenticate/fusion` (`backend/api/fusion.py`, `backend/services/authentication.py`). |

```text
   USER chooses enrolled factors  (face, voice or both)
                    |
     submitted modalities  --compare-->  User Enrollment Profile
                    |
     any submitted modality NOT enrolled?
          /                          \
        yes                           no
         |                             |
  ENROLLMENT_REQUIRED (409)     authenticate EXACTLY the submitted modalities
  that modality is NOT          against the ACTIVE template set  -> fuse (same engine,
  authenticated, nothing        whatever the combination) -> one decision
  is decoded or evaluated                    /                \
  audited as its own state         ACCESS_GRANTED         ACCESS_DENIED
```

The building id, if given, is only recorded in the audit log: the same enrollment and the same submitted factors give the
same decision at every building (`tests/test_flexible_auth.py::test_no_building_specific_modality_rules_remain`).

### Authentication state diagram

```text
   Select factors ---> Submitted modalities enrolled?
   (user's choice)          |                    \
                            | yes                 \ no
                            v                      v
                       Authenticate          ENROLLMENT_REQUIRED   HTTP 409, nothing verified
                       (fusion over exactly       (go to enrollment / pick other factors)
                        the submitted modalities)
                          /          \
             all verified              at least one failed
                  |                          |
            ACCESS_GRANTED             ACCESS_DENIED         (HTTP 200, `authentication_state` in the body)
```

The three outcomes are separate everywhere - backend response, audit log (`authentication_state`), frontend screens,
analytics counters and tests. **A missing enrollment is never an authentication failure**: an `ENROLLMENT_REQUIRED`
attempt has no similarities, no thresholds and no template set in its audit row, and does not count as a denial. The same
rule applies to `/authenticate` and `/verify/*` (single modality).

### Enrollment

Each modality is independent; any subset can be enrolled, in any order. A template set contains templates **only for the
modalities enrolled at the time**. Enrolling a modality later adds its templates to each live set without regenerating
the others (their stored bytes are unchanged), and there is still exactly one ACTIVE set. Re-enrolling a modality replaces
its templates inside the live sets.

Per-modality status (`GET /user/{id}/enrollment-status` -> `statuses`): `NOT_REGISTERED`, `REGISTERED`, `UPDATED`
(re-enrolled), `RETRY_REQUIRED` (voice only: the last enrollment failed the two-recording check and stored nothing;
remembered in `enrollment_events`, cleared by the next success).

- **Face - one-time five-pose enrollment with a centroid.** The user captures Front, Left, Right, Slight Up and Slight Down, one image per pose
  (`pose_front` ... `pose_down` in `POST /enroll`). Per pose: MTCNN detection, the existing alignment, one 512-d FaceNet embedding. Only a capture
  with **no detectable face** or a **blurry** aligned crop (Laplacian variance < 25; measured: normal captures 116-920, visibly blurred <= 34) is
  rejected - nothing else about it is judged. The valid embeddings are averaged and L2-normalized into a **centroid**
  (`embeddings/centroid.py`); the temporary embeddings are discarded at once; T1-T4 are generated **from the centroid alone** through the unchanged
  HKDF + BioHash path; only protected templates are stored. At least 3 valid poses are required (fewer -> 422, nothing stored). The FaceNet model,
  thresholds, BioHash and HKDF are untouched.

  Authentication is unchanged and stays a single capture: one live image -> existing preprocessing -> one embedding -> compared with the ACTIVE
  template (which was generated from the centroid). There are no head-turn prompts during authentication.

  *What is and is not shown:* the centroid sits in the middle of the person's pose cloud, which is what should make a later single capture robust
  to normal appearance changes. On the only data available offline - one portrait with rotations, shifts and exposure changes standing in for
  poses - centroid and single-image templates behaved alike (mean Hamming similarity 0.926 vs 0.929 over eight fresh captures, both 8/8 above the
  0.90 threshold). Several captures land within 0.01 of the threshold, so treat individual pass/fail results near 0.90 as noisy. Real head
  turns across real users have not been evaluated; that is the measurement needed to confirm the benefit.
- **Voice - two recordings, graded check.** The phrase "Security authentication for government access." is recorded twice (4-5 s each)
  and both go in one `POST /enroll` (`image` + `confirm_image`). Before storing anything the backend compares the two by the **cosine
  similarity of their ECAPA-TDNN embeddings** (never waveforms / mel spectrograms) and grades it. The check is deliberately lenient:

  | ECAPA embedding cosine of the two recordings | Band | Result |
  |---|---|---|
  | >= 0.85 | **Excellent** | enrolled |
  | 0.75 - 0.84 | **Good** | enrolled |
  | 0.60 - 0.74 | **Fair** | `409 LOW_QUALITY_WARNING` - nothing stored unless the user continues (`accept_low_quality=true`) or re-records |
  | < 0.60 | **Poor** | `422 ENROLLMENT_INCONSISTENT` - rejected, nothing stored |

  Only Poor is rejected (`RETRY_REQUIRED`, nothing stored). Fair asks the user - "Continue Enrollment" or "Re-record" - and stores
  nothing until they continue. Measured with the real model on TTS speech: the same speaker at different paces 0.98-0.99, with moderate
  microphone noise 0.66-0.73 (Fair), a different speaker 0.51-0.56 (Poor). Pure tones score ~0.99 against each other (they are not
  speech), so this check is not a liveness or spoof test.

### Authentication engine, step by step (`POST /authenticate/fusion`)

1. `fusion_policy` valid; at least one of `face_image` / `voice_audio` (the user's selection) -> else 422;
   an optional `building_id` must exist -> else 404.
2. Compare the submitted modalities with the enrollment profile. Any not enrolled -> **409 `ENROLLMENT_REQUIRED`** with
   `missing_modalities`; audited; nothing decoded or evaluated.
3. Authenticate exactly the submitted modalities against the ACTIVE set; verify all came from the same set.
4. Fuse over exactly those modalities (face only -> the face similarity; voice only -> the voice similarity; face + voice -> both).
5. Return one decision: `ACCESS_GRANTED` / `ACCESS_DENIED`.
6. After `ACCESS_DENIED`, decide the response (next section): `RETRY_REQUESTED` or `SUSPICIOUS_ATTEMPT`. Only the latter
   reaches the security response (template rotation).

### Failure handling: retry vs suspicious escalation (`backend/services/failure_policy.py`)

A biometric mismatch is not evidence of an attack: a genuine user fails a modality because of a poor capture, head pose,
microphone noise or ordinary speech variation. The system therefore separates *"the input was not usable"* from *"the
input was usable but repeatedly failed verification"*, and does not claim to infer intent from a single mismatch.

```text
   capture ---> usable?  -- no -->  CAPTURE_ERROR / QUALITY_INSUFFICIENT  --> RETRY   (never counted)
                  | yes
                  v
             biometric match ---> fusion policy (unchanged) ---> GRANTED -> accept (the mismatch count resets)
                                                 |
                                               DENIED
                                                 |
                        high-quality mismatch count of a modality (consecutive, same ACTIVE set, within the window)
                             <= MAX_MODALITY_RETRIES                 > MAX_MODALITY_RETRIES
                                    |                                         |
                              RETRY_REQUESTED                         SUSPICIOUS_ATTEMPT
                           (no template change)             existing security response -> template rotation
```

| Per-modality status | Meaning | Counts towards escalation? |
|---|---|---|
| `VERIFIED` | passed its threshold | no (resets that modality's count) |
| `VERIFICATION_MISMATCH` | usable capture, did not match | **yes** |
| `QUALITY_INSUFFICIENT` | face: did not match **and** fails the existing enrollment quality gates (confidence, blur, size, centring, roll/yaw); voice: a sentence failed the capture-quality gate and was never compared (docs/VOICE_MODEL.md) | no |
| `CAPTURE_ERROR` | could not be processed: no face detected, undecodable image/audio, silent or < 25 ms recording | no |

- **Capture failure ≠ attack; isolated mismatch ≠ attack.** Both return `RETRY_REQUESTED` with the modalities to capture
  again (`attempt_assessment.retry_modalities`) and change no template, key version or set status.
- **Repeated high-quality mismatch → suspicious escalation.** With `MAX_MODALITY_RETRIES=2` (default) the first and second
  consecutive mismatch of a modality are retried and the third is a `SUSPICIOUS_ATTEMPT`. The count is read from the
  audit log (`audit_logs.modality_statuses`), per modality, and stops at a successful authentication, a rotation, an
  attempt on a different template set (so rotation and manual revocation restart it) or the time window
  (`SUSPICIOUS_MISMATCH_WINDOW_SECONDS`, default 900). Capture/quality failures neither count nor reset.
- **Suspicious escalation → existing security response → template rotation.** The escalation calls the unchanged
  `rotate_after_failed_authentication` (compare-and-swap, one rotation per set, POOL_EXHAUSTED / DISABLED as before). No
  other path changes the template lifecycle.
- **The fusion policy is untouched.** ALL_REQUIRED / WEIGHTED still decide GRANTED / DENIED. A retry
  presents all selected factors again in one attempt; a verified factor is **not** carried over to the next attempt. A
  modality with a `CAPTURE_ERROR` has no score, so that attempt is denied without fusion (fail closed) - previously the
  whole request was a bare, unaudited 422.
- **Wording.** "Authentication incomplete - voice verification requires another attempt" for a retry; "suspicious
  verification pattern detected - repeated high-quality voice verification mismatch" for an escalation. Never "attacker"
  or "attack detected": there is no presentation-attack detection in this system.
- **Audit.** Every attempt records `modality_statuses` and `attempt_status` (`ACCESS_GRANTED` / `RETRY_REQUESTED` /
  `SUSPICIOUS_ATTEMPT`) next to `template_rotation_status` (`NOT_APPLICABLE` / `NOT_TRIGGERED` / `ROTATED` / ...).
  Status names only - no images, audio, embeddings, templates or keys.
- **Voice** recordings pass a capture-quality gate first (silence, clipping, too little speech, severe noise; thresholds
  measured in docs/VOICE_MODEL.md); a recording that passes and does not match is a `VERIFICATION_MISMATCH`. Voice is
  enrolled from two sentences (one combined 192-D centroid) and verified from ONE sentence (protocol change
  2026-09-27; `voice_audio_2` is refused at authentication).
- **Attempt numbering (default).** Consecutive high-quality mismatch 1 of 3 -> `RETRY_REQUESTED` ("Voice verification did
  not match. Please try again."); 2 of 3 -> `RETRY_REQUESTED`; 3 of 3 -> `SUSPICIOUS_ATTEMPT` ("Authentication failed —
  suspicious verification pattern detected." / "Security response: cancellable template rotation triggered."). The
  response carries `mismatch_attempt`, `mismatch_attempt_limit` and an `attempt_notice` sentence.
- **Repeated unusable captures -> cooldown, never rotation.** Capture/quality failures do not reach the mismatch
  counter, so on their own they could otherwise be repeated indefinitely. They are bounded separately: after
  `MAX_CONSECUTIVE_CAPTURE_FAILURES` (default 5) consecutive attempts with an unusable capture within the window, further
  attempts are refused with HTTP 429 `CAPTURE_COOLDOWN` (`Retry-After`) for `CAPTURE_FAILURE_COOLDOWN_SECONDS`
  (default 60), then one attempt per cooldown until a capture is usable or authentication succeeds. The refusal is
  audited (`attempt_status = CAPTURE_COOLDOWN`), evaluates nothing and changes no template. `0` disables it.
- **Remaining limitation.** The cooldown is per user and derived from the audit log; it is not an IP- or device-level
  rate limit, and it does not stop an unlimited number of distinct attempts spread out slower than the cooldown. General
  brute-force protection (lockout, rate limiting at the gateway) and presentation-attack detection are out of scope.
- `MAX_MODALITY_RETRIES`, `MAX_CONSECUTIVE_CAPTURE_FAILURES`, the cooldown and the window are **policy values, not
  derived from data** - review them per deployment.

## 2b. System overview

```text
 REGISTRATION   (each modality is enrolled in turn; every set ends up holding all of them)

  Face capture ---> embedding (once) --+--> for set version v = 1..4:
  Voice capture --> embedding (once) --+       HKDF key v (per modality) -> BioHash -> 256-bit template
                                                              |
                        Template Set 1  [Face V1 | Voice V1]   ACTIVE
                        Template Set 2  [Face V2 | Voice V2]   STANDBY
                        Template Set 3  [Face V3 | Voice V3]   STANDBY
                        Template Set 4  [Face V4 | Voice V4]   STANDBY
                                                              |
                     protected_templates table (bytes + lifecycle metadata only; embeddings discarded)


 AUTHENTICATION

  read the ACTIVE set --> for each submitted modality:
        embedding --> HKDF(active key version) --> BioHash --> candidate --Hamming--> stored template of THAT set
                                                                                 |
                                           per-modality similarity (INTERNAL: audit / debug only)
                                                                                 |
                                         fusion policy --> ONE fusion similarity + decision
                                                                                 |
                                              ACCESS GRANTED  /  ACCESS DENIED
```

## 3. Registration flow

`POST /enroll` (one call per modality) -> `ModalityService.enroll`
(`backend/services/base_service.py`):

1. Validate, decode, preprocess; compute the embedding **once**.
2. `crud.plan_enrollment_sets` picks the target sets: the user's existing live
   (ACTIVE + STANDBY) sets, or - for a new user - `TEMPLATE_POOL_SIZE` (default 4)
   brand-new sets.
3. For each target set, derive the HKDF key (next unused key version for this
   modality) and generate the BioHash template.
4. `crud.save_modality_templates` writes them in one transaction. A new user gets
   set 1 ACTIVE and sets 2-4 STANDBY. A modality enrolled later joins every live
   set; re-enrolling a modality replaces its templates inside the live sets
   (the old rows become REVOKED, `re-enrolled`) and never reuses a key version.
5. The embedding is deleted (`try/finally`) as soon as the templates exist.

After face and voice are both enrolled, every live set holds both.
Response: `templates_created` (sets written to), `active_template_set_version`,
`standby_template_set_versions` - never template bytes.

## 4. Template set lifecycle

```text
                    enroll / generate
                           |
                           v
                       STANDBY  -------------------------.
                           |  promote (revoke / activate)  |
                           v                               |
                        ACTIVE  ---- revoke / superseded --+--> REVOKED   (terminal)
```

| Status | Used to authenticate? |
|---|---|
| `ACTIVE` - exactly one per (user, application) | **Yes - the only one** |
| `STANDBY` | Never |
| `REVOKED` | Never |

## 5. Revocation of a complete multimodal credential

`POST /revoke-template` (multipart: `user_id`, optional `application_id`,
`reason`, and the authorization captures below):

1. **Authorize** (section 8): the caller must authenticate against the ACTIVE set.
   Failure -> `403`, nothing changes.
2. No STANDBY set left -> `409 "Template set pool exhausted. Re-enrollment required."`;
   nothing changes (the ACTIVE set keeps working).
3. Otherwise, in one transaction: every template of the ACTIVE set -> REVOKED
   (`template_set_revoked_at`), and every template of the oldest STANDBY set ->
   ACTIVE (`template_set_activated_at`). Face V2 + Voice V2
   become active simultaneously.

Response: `revoked_template_set_version`, `new_active_template_set_version`,
`remaining_standby_template_sets`. The next authentication regenerates every
candidate under the new set's key versions, so the same person keeps
authenticating. Revoked and new templates are unlinkable (Experiment: 0.504 Hamming similarity).

`POST /templates/{user_id}/activate/{version}` promotes a specific STANDBY set
(previous ACTIVE set -> REVOKED, never demoted to STANDBY).
`POST /templates/{user_id}/generate` adds **one** complete new STANDBY set
(version = highest ever + 1; every modality of the ACTIVE set; fresh key
versions) built from the very captures that authorized the request, and refuses
(`409`) when the pool already holds `TEMPLATE_POOL_SIZE` live sets.

## 6. Database schema

`protected_templates` (one row per template; a set = the rows sharing
`(user_id, application_id, template_set_version)`):

| Column | Notes |
|---|---|
| `template_id`, `user_id`, `modality`, `application_id` | `modality` kept; each row belongs to exactly one set |
| `protected_template`, `output_bits` (256), `key_version` | template bytes never leave the backend |
| `template_version` | BioHash *format* version (stays 1) - **not** the set version |
| `created_at`, `is_active` | `is_active` = the row is in the ACTIVE set |
| `template_status`, `activation_time`, `revoked_time`, `revoked_reason`, `template_group_id` | row-level lifecycle; follows the set (a row replaced by re-enrolling one modality is REVOKED while its set stays ACTIVE) |
| **`template_set_version`** | 1..N, per (user, application), never reused |
| **`template_set_status`** | `ACTIVE` / `STANDBY` / `REVOKED` - set-level, identical on every row of the set |
| **`template_set_created_at`**, **`template_set_activated_at`**, **`template_set_revoked_at`** | set lifecycle times |
| `template_index` | legacy mirror of `template_set_version` |

Constraints: partial unique index `(user_id, modality, application_id) WHERE is_active`
(one ACTIVE row per modality) and `(user_id, application_id, template_set_version, modality)
WHERE template_status <> 'REVOKED'` (one live template per modality per set).

**Honest note on "exactly one ACTIVE set".** With rows-per-template (no separate
set table, by design), the database itself guarantees one active *row per
modality*. That all active rows belong to the *same* set is guaranteed by the
transactional set-level operations in `backend/database/crud.py` and re-verified
before every authentication by `exactly_one_active_set` (section 9), which fails
closed.

`audit_logs` adds `authentication_state` (`ACCESS_GRANTED` / `ACCESS_DENIED` / `ENROLLMENT_REQUIRED`),
`building_id` (already present), `submitted_modalities`, `enrolled_modalities`, `authenticated_modalities`,
`template_set_version`, `template_set_status`, and (from the
previous phase) `face_similarity`, `voice_similarity`,
`fusion_similarity`; `similarity_scores`, `thresholds_used`, `latency_ms`,
`template_versions` (now set versions), `key_versions` are kept. Management
authorization attempts are audited too (`fusion_policy = TEMPLATE_MANAGEMENT`). A small `enrollment_events` table records
whether each modality's enrollment attempts succeeded (`ENROLLED`) or were rejected (`INCONSISTENT`) - metadata only.

## 7. Public API - one fusion similarity

`/authenticate`, `/verify/*` and `/authenticate/fusion` return the same shape:

```json
{
  "user_id": "U001",
  "authentication_state": "ACCESS_GRANTED",
  "status": "ACCESS_GRANTED",
  "authenticated": true,
  "fusion_similarity": 0.981,
  "fusion_distance": 0.019,
  "fusion_threshold": 0.9,
  "fusion_policy": "ALL_REQUIRED",
  "matched_modalities": ["face", "voice"],
  "modalities_used": ["face", "voice"],
  "active_template_set": 2,
  "template_set_version": 2,
  "key_version": 2,
  "authentication_time_ms": 1840,
  "building_id": "national_data_center"
}
```

The values the user needs: `fusion_similarity`, `fusion_distance` (`1 - fusion_similarity`, computed after fusion),
`fusion_threshold`, `authentication_state`, `matched_modalities`, `active_template_set` and `key_version` (the highest among
the modalities' templates in that set). `status` and `template_set_version` repeat `authentication_state` /
`active_template_set` for backward compatibility; `user_id`, `modalities_used`, `authentication_time_ms` and the optional
`building_id` label are metadata. When a submitted modality is not enrolled the endpoint answers **409** instead:

```json
{
  "status": "ENROLLMENT_REQUIRED",
  "authentication_state": "ENROLLMENT_REQUIRED",
  "detail": "Voice is not registered. Complete voice enrollment before authenticating with it.",
  "user_id": "U001",
  "building_id": "national_data_center",
  "submitted_modalities": ["face", "voice"],
  "enrolled_modalities": ["face"],
  "missing_modalities": ["voice"]
}
```

**Absent in production:** the individual modality values - `face_similarity` /
`voice_similarity`, per-modality distances and thresholds, `results`, `fused_score`. With `DEBUG_SCORES=true` they are added to
responses (and to the audit API). They are written to the audit table regardless.

Fusion (equal weights; `s_m` per-modality similarity, `t_m` its threshold):
`fusion_similarity = mean(s_m)`, `fusion_threshold = mean(t_m)`. `authenticated` is
decided by the policy: `ALL_REQUIRED` (default) needs every submitted modality to
pass its own threshold, `WEIGHTED` compares
`fusion_similarity` with `fusion_threshold` (with a per-modality floor). Under
`ALL_REQUIRED` the fusion similarity is informational - it can exceed the
threshold while access is denied - which is why **the result screen shows no
similarity on a denial**, only "Authentication Failed", the reason and the matched
modalities. (The API still returns `fusion_similarity` on denials; the UI hides it.)

## 8. Security: biometric authorization for management

There is no login layer. Revoking, activating or generating a set instead
requires **authenticating against the ACTIVE set**
(`backend/services/template_sets.py::authorize_with_active_set`):

- the request must carry a capture of **every modality the ACTIVE set contains**
  (`face_image`, `voice_audio`) - a missing or
  extra modality -> `403`;
- each capture is embedded and compared with that modality's ACTIVE-set template;
  **all** must pass, else `403 "Biometric authorization failed."` (which modality
  failed is not revealed);
- every attempt, granted or refused, is written to the audit log.

Effect: knowing a `user_id` is not enough to exhaust a victim's set pool (each
refused attempt changes nothing), activate a chosen set, or seed a new standby set
made from an attacker's own biometric. The 403 gate is only as strong as the
biometric match itself - so this is a deterrent, not a substitute for real
authentication. `POST /enroll` on an already-enrolled user still replaces
templates without this check (pre-existing behaviour) and `DELETE /user/{id}`
remains unauthenticated.

**Runtime validation** (`backend/security_validation.py`, before every
comparison, fails closed with HTTP 500): master secret loaded; HKDF seeds valid;
candidate length; stored row usable; `exactly_one_active_set` (all active rows in one
ACTIVE set, `stored` among them, one row per modality - so authentication never
mixes sets); `revoked_never_active`; `standby_unique`; `pool_size_correct`
(<= `TEMPLATE_POOL_SIZE` live sets); `set_status_consistent`;
`key_version_matches_template`. `backend/services/authentication.py` additionally
refuses if the modalities of one request report different set versions.

Limits (unchanged in kind): everything rests on `MASTER_SECRET` (all keys derive
from it) and on the embeddings not leaking - revocation rotates keys, it cannot
change the person; presentation attacks are out of scope; non-invertibility is an
information-loss argument, not a proof (`docs/TEMPLATE_PROTECTION.md`).

## 9. Migration of previously enrolled users

Automatic and idempotent (`backend/database/migration.py::upgrade_schema`, run on
every startup; also `python scripts/migrate_to_multi_template.py [--report]`):

1. `ALTER TABLE ... ADD COLUMN` for the new columns; create the set index.
2. **Pre-pool databases** (one active template per modality): all of a user's active
   templates become **set 1 (ACTIVE)**; old deactivated rows -> REVOKED.
3. **Per-modality pools of the previous phase** (T1..TN per modality): `T-index` becomes the
   set version. If modalities had drifted apart (one revoked further than another) the
   ACTIVE set is the highest version any modality was active at; each modality's row at
   that version becomes ACTIVE, its older live rows are REVOKED
   (`migrated to set-level activation`), newer ones STANDBY.
4. STANDBY sets **cannot** be back-filled for a pre-pool user (they need the
   embeddings, which were never stored). Such a user has a pool of one set - their
   existing templates keep authenticating - until they call `generate` (authorized
   with a capture that matches the ACTIVE set) or re-enroll. Until then
   `revoke-template` returns 409.

Key versions continue after the highest legacy one. `output_bits` is per row, so old
128-bit templates still compare at 128 bits. Verified on a copy of the project's
real legacy database and by tests for both legacy shapes.

## 10. Evaluation

`python -m evaluation.template_set_experiments` writes
`evaluation/results/template_set_{diversity,revocation,promotion,exhaustion}.csv`.
Synthetic embeddings (face 512-d, voice 192-d), 256-bit
templates, 4 sets, two modalities per set, genuine re-capture simulated as cosine
~0.995. The lifecycle runs on the real `crud` + `ModalityService` code over an
in-memory database. (The 403 authorization gate is covered by
`tests/test_template_sets.py`.)

| Experiment | Result |
|---|---|
| Template set diversity | same modality, different sets: mean Hamming similarity 0.503, std 0.032, min 0.383, max 0.594 (240 pairs); same key + genuine re-capture: 0.978 |
| Template set revocation | genuine similarity 0.979 before -> 0.980 after; revoked-vs-new 0.502; both modalities in the new set for 20/20 users |
| Template set promotion | 80/80 genuine attempts accepted across sets 1-4 (~0.98 in every set); 0 attempts mixed sets |
| Template set exhaustion | 3 revocations succeed, the 4th is refused (409); the last ACTIVE set (4) still authenticates |

These test the template mathematics and lifecycle, **not** biometric accuracy.

## 11. Frontend

Buildings come from `GET /buildings`; nothing in React says which factors a facility needs, and no sentence like "this
building requires ..." exists. The user's enrollment profile comes from `GET /user/{id}/enrollment-status`.

- **Building page:** the user's enrollment status per factor (Not registered / Registered / Updated / Retry required) and
  "Select biometric factors for this authentication session." **Begin Authentication** is available once at least one factor
  is enrolled; otherwise **Enroll Biometrics**.
- **Registration:** two independent cards. *Face* - a one-time 5-step guided experience with a circular face guide (Front, Left, Right, Slight
  Up, Slight Down); each capture is checked immediately by the backend (`/enroll/face/check-pose`) and a blurry or faceless one is retaken on the
  spot; the five poses are then sent in one request. *Voice* - the phrase recorded twice, graded by the ECAPA
  embedding cosine: Recording Quality Excellent / Good (green), Fair (yellow, Continue Enrollment / Re-record), Poor (red, Record Again, nothing
  stored). Each card offers Enroll / Update Enrollment / Re-enroll.
- **Authentication:** both factors as selectable cards. Enrolled factors can be selected; a factor that is not registered
  is disabled with an "Enroll" link. Capture cards appear only for the selected factors; the backend authenticates exactly those.
- **Result:** three states. *Access Granted* - building, fusion similarity, matched modalities, active template set, policy,
  time. *Access Denied* - the reason plus presented / verified factors, **no similarity**. *Enrollment Required* - its own screen
  ("You selected Voice, which is not registered ...") with **Go to Enrollment** and **Choose Other Factors**; no processing animation.
- **Template Management (`/templates`):** a pool matrix (Face / Voice x T1..T4 with ACTIVE / STANDBY / REVOKED),
  the set cards, and Activate / Revoke / Generate behind biometric authorization.
- **Analytics:** Success Rate / Denied count only evaluated attempts; Enrollment Required is its own counter.

The frontend types have no field for per-modality scores.

## 12. Viva explanation (2 minutes)

*"A user's credential is not a single template - it is a pool of template sets. At
enrollment we compute each modality's embedding once and derive, from each, four
templates under four different HKDF key versions. Template set 1 is face V1 and voice
V1 - one complete multimodal credential - and it is the only ACTIVE set; sets
2 to 4 are standby. Authentication only ever uses the active set and never mixes templates
from different sets; a runtime check enforces that and fails closed. If a credential is
compromised we revoke the whole active set and promote the next standby set, so face
and voice change together, with no re-capture - and because each key gives a
statistically unrelated template (0.50 Hamming similarity) the revoked set reveals
nothing about its replacement. When the pool is empty the server answers 409 and the
user re-enrolls. Since there is no login, every management action first authenticates a
fresh capture against the active set, so a stranger can't burn through a victim's sets.
The API returns one fused similarity and a decision; per-modality scores live only in the
audit table for testing."*

Likely questions: **"Why sets rather than per-modality pools?"** Per-modality pools let
face and voice drift to different key versions - unrelated credentials to manage,
audit and revoke. A set is one revocable credential, so revocation is atomic and
authentication can prove all modalities come from the same generation. **"Are standby sets
extra secrets?"** No - they are pre-computed under keys derivable from the master secret
anyway; their value is that activation needs no capture. **"What if an embedding leaks?"**
Every set derived from it is compromised and revocation doesn't help - the limit of
cancelable biometrics. **"Why does the denied screen hide the similarity?"** Under
ALL_REQUIRED one failed modality denies access however well the others matched, so a high
number beside a denial is misleading.
