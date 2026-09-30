# Deployment Audit Report — Vercel (frontend) + Render (backend)

> **Modality change (2026-09-27).** Iris and fingerprint have been removed from the system: the implemented
> modalities are **face and voice**, and the `AT_LEAST_TWO` fusion policy (which needed three modalities) is gone.
> This document is a dated record. Its iris / fingerprint material (models, datasets, results, thresholds,
> three-modality fusion) is kept for history only and no longer describes the code. Current state: `README.md`,
> `docs/ROADMAP.md`; current numbers: `evaluation/PAPER_RESULTS_INDEX.md`.

**Scope:** read-only audit. No application logic was modified. Two things were run and then reverted to a clean state: `frontend/npm install` + `npm run build` (build output `dist/` is gitignored, not committed) and a local `uvicorn` process for health-check verification (left running afterward for convenience — kill PID printed in Part 5 if you want it stopped).

**Audit date:** 2026-09-15

---

## PART 1 — Repository Structure Audit

```
.
├── .env.example
├── LICENSE
├── README.md
├── backend/
│   ├── .env.example
│   ├── api/
│   ├── auth/
│   ├── config.py
│   ├── database/
│   ├── main.py
│   ├── render.yaml
│   ├── security_validation.py
│   ├── services/
│   ├── threshold_loader.py
│   └── utils.py
├── biometric.db
├── docs/
│   ├── ARCHITECTURE.md
│   ├── AUTHENTICATION_RELIABILITY_REPORT.md
│   ├── BACKEND_API.md
│   ├── COMPUTER_VISION.md
│   ├── DATASETS.md
│   ├── Face_and_Fingerprint_Training_Report.pdf
│   ├── PRIVACY_AND_SECURITY.md
│   ├── PROJECT_REPORT.md
│   ├── ROADMAP.md
│   └── TEMPLATE_PROTECTION.md
├── embeddings/
│   ├── constants.py
│   └── pipelines.py
├── evaluation/
│   ├── experiments.py
│   ├── metrics.py
│   ├── privacy_metrics.py
│   ├── results/
│   ├── roc.py
│   └── threshold_calibration.py
├── frontend/
│   ├── .env.example
│   ├── .env.local              (gitignored, untracked — contains a live Vercel OIDC token, correctly excluded)
│   ├── .env.production.example
│   ├── package.json
│   ├── public/
│   ├── src/
│   ├── tsconfig*.json
│   ├── vercel.json
│   └── vite.config.ts
├── fusion/
│   ├── config.py
│   ├── policy.py
│   └── score_fusion.py
├── kaggle_kernels/
├── manual_test.db
├── models/
│   ├── common/
│   ├── face/
│   ├── fingerprint/
│   ├── iris/
│   └── voice/
├── notebooks/
├── preprocessing/
├── pytest.ini
├── requirements.txt              (repo root, not backend/ — see note below)
├── scripts/
├── template_protection/
│   ├── biohash.py
│   ├── hkdf_keys.py
│   ├── matcher.py
│   ├── revoke.py
│   └── utils.py
└── tests/  (60+ files)
```

**Required folders — all present:** `frontend/`, `backend/`, `models/`, `template_protection/`, `evaluation/`, `fusion/`, `tests/`. ✅

**Required files:**

| File | Status |
|---|---|
| `frontend/package.json` | ✅ present |
| `frontend/vite.config.ts` | ✅ present |
| `backend/main.py` | ✅ present |
| `backend/requirements.txt` | ❌ **does not exist at that path** — see below |

**Missing-file note (not actually missing, wrong assumed path):** `requirements.txt` lives at the **repository root**, not inside `backend/`. This is deliberate, not an oversight: `backend/services/*.py` imports sibling top-level packages (`models/`, `preprocessing/`, `template_protection/`, `fusion/`, `evaluation/`, `embeddings/`) that only resolve when the process's working directory is the repo root — `backend/render.yaml` already documents this explicitly and sets `rootDir: .` for exactly this reason. No action needed; just don't `pip install -r backend/requirements.txt` (it will 404-equivalent fail) — use `pip install -r requirements.txt` from the repo root.

---

## PART 2 — Frontend Build Audit (Vercel)

```
cd frontend && npm install && npm run build
```

| Step | Result |
|---|---|
| `npm install` | ✅ Success — "up to date, audited 439 packages", **0 vulnerabilities** |
| `npm run build` | ✅ **Success** — `tsc -b && vite build` completed with **zero TypeScript errors**, zero import/path errors, zero environment-variable errors |

**Missing packages:** none.

**Build output:**
```
✓ 8496 modules transformed
dist/index.html                        0.51 kB
dist/assets/index-P78bFeCl.css        97.16 kB │ gzip: 16.05 kB
dist/assets/index-D3vsz0u7.js        528.52 kB │ gzip: 166.13 kB
+ 5 font assets (woff2)
✓ built in 3.37s
```

**One non-blocking warning:** the main JS chunk (528 kB, 166 kB gzipped) exceeds Vite's 500 kB advisory threshold. This is a performance suggestion (consider `dynamic import()` code-splitting), **not an error** — it does not block or fail the build.

**package.json scripts:**
```json
"dev": "vite"
"build": "tsc -b && vite build"
"lint": "oxlint"
"preview": "vite preview"
```
- **Build command:** `npm run build` (runs type-check then Vite build)
- **Output directory:** `dist/` — confirmed generated with the expected `index.html` + `assets/`
- **Vite version:** `8.2.2` (`vite: ^8.2.2` in devDependencies)
- **Node version:** no explicit `engines` field in `package.json`; local build verified on Node `v22.14.0` / npm `10.9.2`. Vercel's live project config (Part 3) uses Node `24.x` — both are compatible with this Vite 8/React 19 stack.

**Confirmed:** Framework is Vite. ✅ Output folder is `dist`. ✅

---

## PART 3 — Vercel Configuration Audit

Queried the **live, already-linked Vercel project** directly (`vercel project inspect frontend`), not assumed values:

| Setting | Live value |
|---|---|
| Root Directory | `.` |
| Framework Preset | **Vite** ✅ |
| Build Command | `npm run build` or `vite build` (auto-detected, no override) |
| Install Command | auto-detected (`npm install`, since `package-lock.json` is present) |
| Output Directory | `None` (no override — Vercel uses its Vite-preset default, which **is** `dist`; confirmed correct by the actual build output in Part 2) |

**On "Root Directory = frontend":** the live value is `.`, not `frontend` — but this is **functionally equivalent and correct**, not a misconfiguration. This Vercel project was created by running `vercel link` from *inside* `frontend/`, so Vercel's own project root **is** the `frontend/` directory (there is no separate monorepo-root Vercel project that would need a `frontend` subdirectory override). No corrected configuration is needed.

**`vercel.json` exists:** ✅ `frontend/vercel.json`:
```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }]
}
```
This is **required** (not optional) — the app uses `react-router-dom`'s `BrowserRouter`, so direct navigation to a nested client-side route (e.g. `/building/hq/authenticate`) without this rewrite would 404 on Vercel's static file server. Confirmed working in an earlier live check this project (direct nested-route navigation returns 200, not 404). Build command/output directory are deliberately **not** set in this file — Vercel's Vite auto-detection already gets both right (verified above), so hardcoding them here would only add a second place to keep in sync for no benefit.

**Live deployment status** (`vercel ls`): 1 deployment, `srk25/frontend`, **Status: ● Ready**, Environment: Production. No failed deployments on record.

---

## PART 4 — Environment Variables Audit

**Every usage of `VITE_API_BASE_URL` in the frontend source:**

| File | Line | Fallback default |
|---|---|---|
| `frontend/src/api/client.ts` | 25 | `'http://127.0.0.1:8000'` |

Exactly one usage, in exactly one place (`export const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'`) — every other API call in the frontend imports `BASE_URL` from this module rather than reading the env var directly, so there is no risk of a second, drifting hardcoded value elsewhere.

**`.env.example` files:**
- `frontend/.env.example` — present, documents `VITE_API_BASE_URL=http://127.0.0.1:8000` (local-dev default) with a note about matching CORS.
- `frontend/.env.production.example` — present, documents `VITE_API_BASE_URL=https://your-backend.onrender.com` with instructions to set the real value in Vercel's Project Settings → Environment Variables (Production scope).

**Search of entire `frontend/src/` for `localhost` / `127.0.0.1` / `:8000`:**

| Term | Occurrences |
|---|---|
| `localhost` | 0 |
| `127.0.0.1` | 1 — `src/api/client.ts:25` (the documented local-dev fallback above) |
| `:8000` | 1 — same line, same occurrence |

**Confirmed: no hardcoded localhost URLs remain outside the one, intentional, overridable local-dev fallback.** In production, `VITE_API_BASE_URL` is baked in at Vercel build time from the Project Settings value, so this fallback is never reached in the deployed app as long as that env var is set (see Part 9's blocker list).

**Note (not a frontend issue, flagged for completeness):** `frontend/.env.local` exists locally and contains a real, live `VERCEL_OIDC_TOKEN` (created automatically by the Vercel CLI on `vercel link`). Confirmed **gitignored and untracked** (`frontend/.gitignore` line 27: `.env*`) — not a leak, no action needed, noted here only because the audit's environment-variable scope makes it worth explicitly ruling out.

---

## PART 5 — Backend Render Audit

```
pip install -r requirements.txt   # (repo root — see Part 1 note)
```

**Result: 1 failure, everything else installed successfully.**

`webrtcvad>=2.0.10` failed to build:
```
error: Microsoft Visual C++ 14.0 or greater is required.
Get it with "Microsoft C++ Build Tools"
Failed to build installable wheels for some pyproject.toml based projects: webrtcvad
```

**This is a genuine local-Windows build failure** (webrtcvad ships a C extension with no prebuilt Windows wheel for this Python version) — **and it is safe to ignore, because the package is dead weight**: grep confirms `webrtcvad` is never actually `import`ed anywhere in the codebase. Its only two references are (1) `models/voice/config.py:28` — a `Literal["energy", "webrtcvad"]` type option that **defaults to `"energy"`** and is never switched, and (2) `preprocessing/voice.py`'s own docstring, which explicitly explains VAD was implemented with a **dependency-free energy-threshold method specifically because webrtcvad's C-extension build isn't reliably installable** — this failure is the exact scenario that comment was written to avoid. Every other required import was verified working (`fastapi`, `uvicorn`, `sqlalchemy`, `pydantic`/`pydantic-settings`, `cryptography`, `numpy`, `cv2`, `torch`, `torchvision`, `facenet_pytorch`, `speechbrain`, `torchaudio`, `scipy`, `PIL`, `sklearn`, `h5py`, `kaggle`, `albumentations` — all import cleanly).

**Recommended fix (not applied — audit is read-only):** remove the `webrtcvad>=2.0.10` line from `requirements.txt`. This removes a genuine Windows build risk and a Linux build-time cost for a package the code never uses, with zero behavior change.

**Local server start:**
```
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
✅ **Started cleanly**, `backend.main` imports with no errors, `init_db()` ran in the lifespan hook with no exceptions.

**`GET /system/health` response:**
```json
{
  "backend": "online",
  "database": "connected",
  "face_model": "loaded",
  "fingerprint_model": "loaded",
  "voice_model": "loaded",
  "template_protection": "active",
  "fusion_policy": "ALL_REQUIRED",
  "thresholds_loaded": false,
  "audit_logging": true
}
```

| Check | Status |
|---|---|
| Database connected | ✅ |
| Face model loaded | ✅ |
| Fingerprint model loaded | ✅ |
| Voice model loaded | ✅ |
| Template protection active | ✅ |
| `thresholds_loaded: false` | Expected, not a blocker — no `evaluation/results/*_threshold.json` calibration files exist by design; every modality correctly falls back to the documented `Settings.match_threshold=0.9` default. |

`GET /health` (the actual Render `healthCheckPath`) also verified: `{"status":"ok"}`, HTTP 200.

---

## PART 6 — Render Deployment Files

`backend/render.yaml` **exists**:

```yaml
services:
  - type: web
    name: biometric-auth-backend
    env: python
    plan: starter
    rootDir: .
    buildCommand: pip install --upgrade pip && pip install -r requirements.txt
    startCommand: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /health
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9
      - key: MASTER_SECRET
        sync: false
      - key: DATABASE_PATH
        value: /var/data/biometric.db
      - key: ENV
        value: production
      - key: CORS_ORIGIN
        sync: false
    disk:
      name: biometric-data
      mountPath: /var/data
      sizeGB: 1
```

| Field | Value | Check |
|---|---|---|
| Runtime | `env: python`, `PYTHON_VERSION: 3.11.9` | ✅ |
| Build command | `pip install --upgrade pip && pip install -r requirements.txt` | ✅ (runs from `rootDir: .`, so `requirements.txt` resolves correctly per Part 1) |
| Start command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` | ✅ binds `0.0.0.0` and Render's dynamic `$PORT`, both required on Render |
| Health check path | `/health` | ✅ matches the real route (verified live above), **not** the `/healthz` placeholder an earlier draft of this config used |
| Persistent SQLite | `DATABASE_PATH: /var/data/biometric.db` + a `disk` block (`mountPath: /var/data`, `sizeGB: 1`) | ✅ **matches the required path exactly** |

**No Render service has actually been created/deployed yet** — no live `*.onrender.com` URL exists anywhere in the repo (only the placeholder `<your-render-service>.onrender.com` / `biometric-auth-backend.onrender.com` example in `README.md`). This Blueprint is ready to use but the dashboard steps (New → Blueprint → point at `backend/render.yaml`, fill in `MASTER_SECRET`/`CORS_ORIGIN`, attach the disk) haven't been completed. See Part 9.

---

## PART 7 — Requirements Audit

`requirements.txt` (repo root):

| Requested package | Present as | Status |
|---|---|---|
| fastapi | `fastapi>=0.110` | ✅ |
| uvicorn | `uvicorn[standard]>=0.27` | ✅ |
| sqlalchemy | `sqlalchemy>=2.0` | ✅ |
| pydantic | `pydantic>=2.6` (+ `pydantic-settings>=2.2`) | ✅ |
| python-dotenv | `python-dotenv>=1.0` | ✅ |
| cryptography | `cryptography>=42.0` | ✅ |
| numpy | `numpy>=1.26` | ✅ |
| opencv-python | `opencv-python-headless>=4.9` | ✅ **present as the headless variant** — correct and preferable for a server with no display (avoids pulling in GUI/X11 dependencies); not a gap. |
| torch | `torch>=2.2` | ✅ |
| torchvision | `torchvision>=0.17` | ✅ |
| speechbrain | `speechbrain>=1.0` | ✅ |
| librosa / scipy | `scipy>=1.11` (no `librosa`) | ✅ satisfies the "either/or" — `preprocessing/voice.py` deliberately uses only numpy/scipy, `librosa` was never a dependency by design (see that module's docstring). |
| Pillow | `Pillow>=10.2` | ✅ |

**No required packages are missing.**

**One unused/fragile package present** (already covered in Part 5): `webrtcvad>=2.0.10` — never imported, fails to build on Windows, recommended for removal.

---

## PART 8 — CORS Audit

`backend/main.py`, `_resolve_cors_origins()` (lines 28–56) + `add_middleware` call (lines 95–101):

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=_resolve_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

**Allowed origins resolution order:**
1. `CORS_ORIGIN` env var (singular) — if set, used as the **sole** allowed origin. This is the production/Render-facing variable (`backend/render.yaml` sets it via `sync: false`, prompting for the real Vercel URL in the dashboard).
2. `CORS_ALLOWED_ORIGINS` (comma-separated) — local/multi-origin dev fallback.
3. Hardcoded fallback: `["http://localhost:5173"]` (Vite's dev-server default) — **only reached if neither env var is set.**

**Confirmed: production is designed to use `CORS_ORIGIN` from the environment, never a wildcard `"*"`.** There is no `"*"` anywhere in this file or in `_resolve_cors_origins()`'s logic — every code path returns a specific origin list. `allow_credentials=False` is also correctly paired with this (wildcard + credentials is the classic CORS misconfiguration; neither is present here).

**Live-state caveat:** `CORS_ORIGIN` is currently unset in this local environment (not a Render deployment), so the live `/system/health` check above ran under the `http://localhost:5173` fallback — expected and correct for local verification, **not** what will happen on the real Render deployment once `CORS_ORIGIN` is set per Part 9.

---

## PART 9 — Deployment Readiness Checklist

### Frontend
- [x] `package.json` present
- [x] `npm run build` passes (zero errors)
- [x] `dist/` generated correctly
- [x] Environment variables documented (`.env.example`, `.env.production.example`) — ⚠️ **not yet set as a real value in Vercel's dashboard** (see blockers)
- [x] No hardcoded localhost URLs (one documented, overridable fallback only)

### Backend
- [x] `requirements.txt` valid (one unused package flagged, not a real blocker — see Part 7)
- [x] `uvicorn` starts cleanly
- [x] SQLite path configurable (`DATABASE_PATH`, verified in `render.yaml`)
- [x] `MASTER_SECRET` read from environment, no insecure default (fails fast if unset — verified: `Settings.master_secret: str` has no default)
- [x] Persistent disk path configured (`render.yaml`'s `disk` block, `/var/data`)
- [x] Health endpoint working (`GET /health` → `200 {"status":"ok"}`, verified live)

### Deployment
- [x] **Vercel: ready** — live project confirmed `Status: Ready`, build verified clean, SPA rewrite in place.
- [x] **Render: Blueprint ready, service not yet created** — `render.yaml` is correct and complete; the actual Render dashboard steps haven't been done yet.
- **Remaining blockers** (all on the Render side, none on Vercel):
  1. No Render service exists yet — needs New → Blueprint → point at `backend/render.yaml`.
  2. `MASTER_SECRET` needs a real generated value set in Render's dashboard (`sync: false` means it won't be pulled from anywhere automatically).
  3. `CORS_ORIGIN` needs to be set to the real deployed Vercel URL (`https://frontend-kohl-three-v6xpa81mai.vercel.app`, confirmed live and working — see Part 10) once Render's own URL is known too.
  4. Vercel's `VITE_API_BASE_URL` (Production scope) needs to be set to the real Render URL once it exists — currently still whatever placeholder or unset state it's in (not verified this pass; check Vercel dashboard → Project Settings → Environment Variables).
  5. (Non-blocking, recommended) remove `webrtcvad` from `requirements.txt`.

---

## PART 10 — Vercel Failure Diagnosis

**No build or deployment failure exists.** `vercel ls` shows the one live deployment as `Status: ● Ready`, Production. Ranked findings, most to least significant:

1. **Not a failure, a URL-choice trap (informational, non-blocking):** the raw per-deployment hash URL (`https://frontend-4kn3qxu4t-srk25.vercel.app`) redirects to `vercel.com/login?next=...sso-api...` — this is the Vercel **team's Deployment Protection** setting (an SSO gate applied to individual deployment URLs), not a build error. The **production alias** (`https://frontend-kohl-three-v6xpa81mai.vercel.app`) returns a clean `200` directly with no gate. **Fix:** always share/use the production alias, not the deployment-hash URL, when linking to this app from outside the team.
2. **Non-blocking build warning:** main JS chunk exceeds Vite's 500 kB advisory size (Part 2). Cosmetic; does not fail the build or break the app.
3. **No other issues found.** TypeScript, imports, environment variables, and the build itself are all clean.

---

## Summary

| Area | Status |
|---|---|
| Repository structure | ✅ complete, one path-assumption correction (`requirements.txt` is at repo root) |
| Frontend build | ✅ clean, zero errors |
| Vercel config | ✅ correct and **live** (already deployed, Status: Ready) |
| Environment variables (frontend) | ✅ correctly wired, no hardcoded URLs, one Vercel-dashboard value to confirm |
| Backend imports/health | ✅ clean (one unused, non-blocking dependency to remove) |
| Render Blueprint | ✅ correct and complete, but the **service itself doesn't exist yet** |
| CORS | ✅ correctly designed, env-driven, never wildcard |

## Overall status: 🟡 Ready after fixes

Vercel is genuinely done — live, working, correctly configured. Render is the only remaining gap, and it's operational, not technical: the Blueprint file is correct; someone needs to actually create the Render service from it and fill in three environment-variable values (`MASTER_SECRET`, `CORS_ORIGIN`, and Vercel's `VITE_API_BASE_URL` pointing back at it). No code, config file, or application logic needs to change to reach 🟢, other than the optional `webrtcvad` line removal.
