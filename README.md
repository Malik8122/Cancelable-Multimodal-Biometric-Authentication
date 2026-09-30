# Cancelable Multimodal Biometric Authentication for Critical Infrastructure

A privacy-preserving biometric authentication system that combines **face
and voice** recognition (the user chooses which enrolled factors to present)
with **cancelable biometric template protection**, so
that no raw biometric image or unprotected biometric embedding is ever stored
as the permanent authentication credential.

> Current, traceable results: [`evaluation/PAPER_RESULTS_INDEX.md`](evaluation/PAPER_RESULTS_INDEX.md);
> project validation: [`docs/FINAL_PROJECT_VALIDATION.md`](docs/FINAL_PROJECT_VALIDATION.md).
> Iris and fingerprint were removed from the system (2026-09-27); the supported modalities are face and voice, plus an
> optional **dynamic hand gesture** (a Z drawn in the air, MediaPipe Hands landmarks + DTW; threshold from synthetic data only) -
> see [`docs/HAND_GESTURE_MODALITY.md`](docs/HAND_GESTURE_MODALITY.md).

This is a final-year capstone project. It is a research/demo system, not a
production-hardened security product — see [Limitations & honest
scope](#limitations--honest-scope) before drawing any stronger conclusion
from it.

**Preparing for a project evaluation/viva?** [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md)
is the full in-depth writeup — motivation, why multimodal, why cancelable
biometrics, how this compares to existing solutions, every model/dataset
decision justified, the real GPU training results (including the honest
ones), and a prepared-answers section for the questions reviewers are most
likely to ask. [`docs/COMPUTER_VISION.md`](docs/COMPUTER_VISION.md) is the
companion deep dive into every computer-vision technique used, with code
references.

## The problem

Traditional biometric systems store templates derived directly from a
person's face or voice. Unlike a password, a biometric
characteristic can't be changed after a breach — so a compromised biometric
database is a permanent, not a recoverable, exposure. Combining several
modalities improves recognition reliability, but naively multiplies the
amount of sensitive biometric data being handled unless the storage layer is
designed around that risk from the start.

## What this project actually is

Not "face recognition + voice recognition."  The
central contribution is combining multimodal biometric recognition **with** a
cancelable, revocable template-protection layer, so that:

```text
Biometric input -> feature extraction -> embedding -> cancelable
transformation -> protected template -> authentication
```

and the protected template — never the raw image, never the raw embedding —
is what gets stored and compared.

## Architecture

Full diagrams and the per-modality classical-CV-vs-deep-learning breakdown
are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Summary:

```text
Face image / voice recording
        |
Modality-specific preprocessing (preprocessing/)
        |
Modality-specific embedding model (models/)
        |
BaseEmbedder.extract_embedding()  <- one shared interface for every modality
        |
Cancelable transformation (template_protection/)        [Phase 2]
        |
Protected template  ->  backend/ (FastAPI + SQLite)      [Phase 2]
        |
Multimodal score fusion (fusion/)                        [Phase 3]
        |
AUTHENTICATE / REJECT
```

### Biometric similarity metrics

| Component | Metric | Threshold / decision |
|---|---|---|
| Face | Cosine similarity (calibrated estimate) | ≥ 0.80, higher is better |
| Voice | Euclidean distance (calibrated estimate) | ≤ 0.75, **lower** is better |
| Hand gesture | Median DTW distance to the 3 enrolled Z samples | ≤ 0.216 (development default from a synthetic calibration; not yet calibrated on real users), **lower** is better |
| Fusion | Mean of the modality scores on one higher-is-better scale (voice converted with 1 − d²/2) | `ALL_REQUIRED` by default |
| Cancelable template | Hamming distance between 256-bit BioHash templates | the comparison the decisions are derived from |

Raw embeddings are never stored, so face cosine and voice Euclidean distance cannot be computed exactly at
verification. They are estimated from the template Hamming comparison through an offline calibration of the BioHash
transform itself (not real biometric data), and are always labelled as estimates. The 0.80 / 0.75 thresholds are
teacher-requested project settings (`FACE_COSINE_THRESHOLD`, `VOICE_EUCLIDEAN_THRESHOLD`), not experimentally
calibrated. Full explanation: [`docs/BIOMETRIC_METRICS.md`](docs/BIOMETRIC_METRICS.md).

Registration asks for a human-readable name first (`POST /users`). The name is only a label: the server-generated
internal `user_id` remains the identifier, so two people may share a name.

## Project roadmap

**Current architecture:** authentication is user-driven. Users enroll face, voice or both and choose which enrolled factors to present in each session; buildings are context only (no biometric policy). The backend authenticates and fuses exactly the submitted modalities - a modality that is not enrolled is `ENROLLMENT_REQUIRED`, not a denial. A user's credential is a pool of *template sets* (set 1 active, the rest standby) revoked as a whole, and only one fused similarity is exposed. See [`docs/MULTI_TEMPLATE_ARCHITECTURE.md`](docs/MULTI_TEMPLATE_ARCHITECTURE.md).

Full detail in [`docs/ROADMAP.md`](docs/ROADMAP.md). Every phase advances all
modalities together — no modality is ever left behind.

| Phase | Focus | Status |
|---|---|---|
| **1** | Preprocessing + recognition models, individually evaluated | ✅ Implemented for face and voice (iris and fingerprint were removed) |
| **2** | Cancelable template protection, key management, backend + storage | ✅ Implemented (this branch) |
| **3** | Multimodal fusion, React dashboard, full cross-modality evaluation | 🔜 Next |

## Repository structure

```text
backend/              FastAPI app: api/, services/, database/, auth/, config.py
models/
  common/             BaseEmbedder interface + ArcFace training head
  face/ voice/         inference.py + saved/ (Git-LFS-tracked checkpoints)
preprocessing/         Modality-specific classical-CV preprocessing
embeddings/            Preprocessing + model wired together behind one interface
template_protection/   Cancelable BioHashing transform, HKDF key management, matcher
fusion/                Multimodal score fusion (Phase 3)
evaluation/            Shared metrics (FAR/FRR/EER/ROC/AUC) + experiment runners,
                       including privacy_metrics.py (protected-template experiments)
frontend/              React dashboard (Phase 3)
notebooks/             Colab notebooks: dataset download, fine-tuning, image testing
kaggle_kernels/         Kaggle Kernel equivalents: unattended real-GPU training via the API
scripts/                run_kaggle_kernels.py: push/monitor/pull the Kaggle kernels
docs/                  Architecture, datasets/licensing, privacy analysis, roadmap,
                       template protection + backend API reference
tests/                 Offline tests against synthetic images (no GPU/dataset needed)
```

## Getting started

### Prerequisites

- Python 3.10+
- [Git LFS](https://git-lfs.com/) (`git lfs install`) — trained checkpoints
  (`.pt` files) are tracked via LFS, not committed as regular blobs.

### Setup

```bash
git clone https://github.com/Malik8122/Cancelable-Multimodal-Biometric-Authentication-for-Critical-Infrastructure.git
cd Cancelable-Multimodal-Biometric-Authentication-for-Critical-Infrastructure
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run the offline test suite

No GPU, no dataset, no trained checkpoint required — these tests exercise the
preprocessing → embedding interface with synthetic images and prove the
pipeline is correctly wired for every modality, plus the full
`template_protection/` + `backend/` layer against synthetic embeddings and an
in-memory database:

```bash
pytest
```

### Configure the backend environment

The backend needs a `MASTER_SECRET` — the root secret every cancelable
template's key is derived from (see `docs/TEMPLATE_PROTECTION.md`). There is
no working default; the server refuses to start without one.

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"   # paste the output into .env as MASTER_SECRET
```

`.env` is gitignored — never commit a real secret. `pytest` doesn't need this
step; the test suite supplies its own obviously-fake secret via a fixture
(`tests/conftest.py`).

### Run the backend

```bash
uvicorn backend.main:app --reload
```

Then see `docs/BACKEND_API.md` for the full endpoint reference, or open
`http://127.0.0.1:8000/docs` for interactive Swagger docs. By default this
creates a local `biometric.db` SQLite file (gitignored) storing **only**
protected templates — never raw images or raw embeddings.

### Run the frontend

Requires Node.js 20.19+ or 22.12+ (Vite 8). In a second terminal:

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 - talks to http://127.0.0.1:8000 by default
```

Open `http://localhost:5173`, pick a building, choose **+ New Registration**, enter a name, then enroll face and
voice. Camera and microphone access need `localhost` or HTTPS. Set `DEBUG_SCORES=true` in `.env` (local development
only) to see Face / Voice Similarity on the result page; otherwise only the fusion score and decision are shown.

### Train the models

Two equivalent ways to run the same training/evaluation/image-testing logic
— pick whichever fits your workflow. Either way, each run: downloads its
modality's dataset (see licensing notes below), preprocesses it with this
repo's own `preprocessing/` code, fine-tunes the embedding model with an
ArcFace head, **saves the resulting checkpoint** (both `.pt` — the format
`models/<modality>/inference.py` actually loads — and a companion `.h5`
interoperability export) to `models/<modality>/saved/`, evaluates it
(accuracy/FAR/FRR/EER/ROC-AUC), and runs an image-based testing section —
genuine vs. impostor pairs and a gallery-matching demo, with the actual
images and similarity scores displayed, not just a metrics table.

**Option A — Google Colab (manual, interactive):** open each notebook below
in Colab (GPU runtime) and run top to bottom — no Google Drive mount
required.

- [`notebooks/01_face_training_and_testing.ipynb`](notebooks/01_face_training_and_testing.ipynb)
- [`notebooks/04_voice_training.ipynb`](notebooks/04_voice_training.ipynb)

Then download the resulting checkpoints and commit them locally.

**Option B — Kaggle Kernels (unattended, scriptable, free GPU):** see
[`kaggle_kernels/README.md`](kaggle_kernels/README.md). Datasets are attached
natively instead of downloaded manually. Once you have a Kaggle API token in place:

```bash
python scripts/run_kaggle_kernels.py
```

pushes the face and voice kernels, waits for each to finish on Kaggle's GPU, pulls the
resulting `.pt`/`.h5` checkpoints, and installs them into
`models/<modality>/saved/` automatically. Then:

```bash
git add models/*/saved/*.pt models/*/saved/*.h5   # Git LFS picks these up automatically
git commit -m "Add trained checkpoints from Kaggle GPU training"
git push
```

Until a checkpoint exists for a modality, `BaseEmbedder` transparently falls
back to a deterministic mock embedding (see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#mock-mode)) so the rest of the
system can still be developed and tested.

## Deployment (Render + Vercel)

Backend on [Render](https://render.com) (with a persistent disk for the
SQLite file), frontend on [Vercel](https://vercel.com). Nothing about
authentication logic, model behavior, API routes, or payloads changes
between local dev and this deployment - it's configuration only.

### Backend -> Render

1. Push this repo to GitHub (Render deploys from a Git connection).
2. In the Render dashboard: **New -> Blueprint**, point it at
   [`backend/render.yaml`](backend/render.yaml), and confirm the service's
   root directory resolves to the **repository root** (not `backend/`) -
   `backend/render.yaml`'s own comments explain why (the backend imports
   sibling packages like `models/` and `template_protection/` that only
   resolve from the repo root).
3. Render provisions a 1GB persistent disk mounted at `/var/data` (see the
   `disk:` block in `backend/render.yaml`) so the SQLite database survives
   deploys/restarts, unlike the container's own ephemeral filesystem.
4. Set these environment variables in the Render dashboard (also documented
   in [`backend/.env.example`](backend/.env.example)):

   | Variable | Value |
   |---|---|
   | `MASTER_SECRET` | A real secret - `python -c "import secrets; print(secrets.token_hex(32))"`. Never reuse a value from any `.env.example`. |
   | `DATABASE_PATH` | `/var/data/biometric.db` (matches the disk mount above) |
   | `ENV` | `production` (disables `/docs`, `/redoc`, `/openapi.json`) |
   | `CORS_ORIGIN` | Your deployed Vercel URL, e.g. `https://your-app.vercel.app` (no trailing slash) - set this *after* step 5 below, once you know the real URL |

5. **Trained checkpoints**: this repo tracks `.pt`/`.h5` files under
   `models/*/saved/` via Git LFS (see [Prerequisites](#prerequisites)).
   Render's build clones your repo including LFS objects automatically for a
   GitHub-connected service; if a checkpoint ever comes back missing (check
   `GET /system/health`'s modality-level fields), `BaseEmbedder` silently
   falls back to a non-biometric mock embedding for that modality - the
   service still runs, but authentication for that modality stops being
   meaningful. Verify LFS objects are present rather than pointer files if
   this happens.
6. Confirm it's up: `curl https://<your-render-service>.onrender.com/health`
   should return `{"status": "ok"}`. `/docs` should now 404 (Swagger is
   disabled in production).

### Frontend -> Vercel

1. Import this repo into Vercel; set its **Root Directory** to `frontend/`
   (Vercel auto-detects the Vite framework preset from there).
2. In Vercel Project Settings -> Environment Variables, add
   `VITE_API_BASE_URL` = your Render backend's URL from the previous
   section (e.g. `https://biometric-auth-backend.onrender.com`), scoped to
   the **Production** environment. [`frontend/.env.production.example`](frontend/.env.production.example)
   documents the same variable for a local production build
   (`npm run build`).
3. Deploy. Vercel serves everything over HTTPS by default, which is a hard
   requirement for `getUserMedia` (camera/microphone capture) to work in any
   real browser outside `localhost` - no extra configuration needed for
   that specifically, just don't introduce a non-HTTPS asset/API URL
   anywhere.
4. Go back to Render and set `CORS_ORIGIN` to this exact Vercel URL, then
   redeploy the backend (env var changes require a restart) - until this is
   set correctly, the browser will block every request with a CORS error,
   and [`BackendStatusBanner`](frontend/src/components/layout/BackendStatusBanner.tsx)
   will show a friendly "backend temporarily unreachable" message rather
   than fail silently or fabricate a connected state.

### Local production build (optional sanity check before deploying)

```bash
# Backend
cp backend/.env.example .env   # from the repo root; fill in a real MASTER_SECRET
ENV=production uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Frontend, in another terminal
cd frontend
cp .env.production.example .env.production   # fill in the backend URL above
npm run build && npm run preview
```

## Model decisions

| Modality | Model | Why |
|---|---|---|
| Face | InceptionResnetV1, pretrained on VGGFace2 (`facenet-pytorch`) | Strong pretrained embeddings, minimal fine-tuning needed |
| Voice | ECAPA-TDNN (SpeechBrain), 192-dim embedding | Current state-of-the-art speaker-verification architecture, PyTorch-compatible, easy to fine-tune; see [`docs/VOICE_MODEL.md`](docs/VOICE_MODEL.md) |

Both are fine-tuned with the same ArcFace angular-margin loss
(`models/common/arcface.py`, used by voice through `models/voice/losses.py`) for a
consistent, comparable training recipe across modalities.

## Datasets

| Modality | Dataset | Access |
|---|---|---|
| Face | [LFW](http://vis-www.cs.umass.edu/lfw/) | Auto-downloaded, no login |
| Voice | [VoxCeleb1 subset (Indian celebrities)](https://www.kaggle.com/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity) | Kaggle account required (free); see [`docs/DATASETS.md`](docs/DATASETS.md) for why this is a subset, not the full corpus |

Full source/license/retention notes: [`docs/DATASETS.md`](docs/DATASETS.md).
**No raw biometric image from any dataset is ever committed to this repo.**

## Privacy design

- Raw biometric samples exist only for the duration of a preprocessing call —
  nothing persists them to disk.
- The database (`backend/database/`) stores **only** cancelable/protected
  templates, never raw images or raw embeddings.
- Transformation keys are derived via HKDF-SHA256
  (`template_protection/hkdf_keys.py`), not stored plaintext beside the
  templates they produced.
- Revocability and cross-application diversity are experimentally
  demonstrated (`evaluation/privacy_metrics.py`), not just asserted.

Full principles, the BioHashing/HKDF design, and current honest limitations:
[`docs/PRIVACY_AND_SECURITY.md`](docs/PRIVACY_AND_SECURITY.md) and
[`docs/TEMPLATE_PROTECTION.md`](docs/TEMPLATE_PROTECTION.md).

## Limitations & honest scope

- This is a capstone research/demo system, **not** a production-grade
  security product. No claim of "100% secure," formal cryptographic
  irreversibility, or regulatory compliance is made anywhere in this
  codebase.
- The cancelable transform's non-invertibility is an information-lossy
  argument, not a cryptographic one-wayness proof — see
  [`docs/TEMPLATE_PROTECTION.md`](docs/TEMPLATE_PROTECTION.md)'s "Security
  assumptions and limitations" for exactly what is and isn't claimed.
- There is no user/API-caller authentication (API keys, OAuth, sessions) —
  every endpoint in `backend/` is reachable by anyone who can reach the
  process; that's out of scope for this capstone's biometric-verification
  focus, not an oversight (see `docs/BACKEND_API.md`).

## Team / contributor roles

- **Member 1 — Face:** dataset, preprocessing, model, evaluation, `models/face/`.
- **Member 2 — Iris** (modality since removed from the system).
- **Member 3 — Fingerprint (modality since removed) + integration/privacy:** the
  `template_protection/` cancelable-transform implementation and the `backend/`
  FastAPI integration.

Every modality shares the `BaseEmbedder` interface
(`models/common/base_embedder.py`) precisely so this split can happen without
one member's work blocking another's.

## License

MIT — see [`LICENSE`](LICENSE). This covers the code in this repository; it
does **not** grant any rights to the third-party datasets referenced in
[`docs/DATASETS.md`](docs/DATASETS.md), which carry their own separate
licenses.
