# Privacy & Security

This document tracks the project's privacy-preserving design and its honest
limitations. Phase 1 established the recognition foundation (raw-data
minimization only); **Phase 2** (`template_protection/` + `backend/`) is
implemented and is what this document now describes in full — see
`docs/TEMPLATE_PROTECTION.md` for the mathematical detail behind every claim
made here.

## Principles (see also `README.md` § Privacy Design)

1. **Raw biometric minimization.** Raw images exist only for the duration of
   a preprocessing/embedding call; nothing in `preprocessing/` or `models/`
   writes a raw image to disk. `.gitignore` blocks committing image files
   outright as a second line of defense.
2. **No unprotected embedding as a credential.** The backend
   (`backend/services/*.py`) only ever persists the output of
   `template_protection.biohash.generate_template`
   (`backend/database/models.py::ProtectedTemplate.protected_template`),
   never a raw `BaseEmbedder.extract_embedding()` output.
3. **Key separation.** Transformation keys are derived via HKDF-SHA256
   (`template_protection/hkdf_keys.py`) from a server-side secret
   (`MASTER_SECRET`, environment-only, see `backend/config.py`) plus a
   user/modality/application identifier — never stored plaintext, or at all,
   next to the protected template they produced.
4. **Revocability & diversity.** Changing the transformation key
   (`template_protection/revoke.py`) produces an unlinkable template for the
   same underlying biometric; the same biometric under different application
   IDs is likewise unlinkable — both measured empirically by
   `evaluation/privacy_metrics.py`'s Experiment 2 and Experiment 3 (expected
   pairwise Hamming distance ≈ 0.5 in both cases).

## What this codebase does NOT claim

- **No cryptographic one-wayness.** The cancelable transform is
  information-lossy (a real-valued embedding collapses to a much shorter bit
  string), which is a data-transformation argument for why reconstruction
  from the template *alone* is infeasible — it is **not** a
  computational-hardness proof like a cryptographic hash function has. If an
  attacker also recovers the key material, partial reconstruction becomes a
  studied attack in the BioHashing literature (see
  `docs/TEMPLATE_PROTECTION.md`'s "Security assumptions and limitations").
- **No claim of regulatory compliance** anywhere in this codebase.
- **Mock-mode templates aren't biometrically meaningful.** When a modality's
  checkpoint is missing (e.g. Git LFS objects not pulled), templates generated from
  `BaseEmbedder`'s mock-mode fallback must not be treated as real
  authentication material outside testing/demos —
  `embeddings.pipelines.ModalityPipeline.is_mock` exposes this so calling
  code (`backend/api/enroll.py`) can flag it.

## Threat/analysis topics (Experiment 5)

- **Template leakage in a database-compromise scenario.** An attacker who
  reads `biometric.db` gets `protected_template` bytes, `key_version`,
  `template_version`, and identifiers — never a raw image, never a raw
  embedding, never `MASTER_SECRET` (which lives only in the environment, not
  the database). Without the key material, the leaked bytes are, by
  Experiment 1's design intent, not directly reversible to the embedding.
- **Replay-attack risk.** Nothing in this API cryptographically binds an
  `/authenticate` request to a live capture (e.g. no liveness detection,
  no challenge-response) — a captured *image* could be replayed. This is an
  acquisition-layer concern the current scope doesn't address; the
  template-protection layer's job is what happens to an image once it's
  captured, not proving the image is live. Documented here rather than
  silently out of scope.
- **Transformation-key compromise.** If `MASTER_SECRET` leaks, every derived
  key for every user/modality/application is recoverable (HKDF is
  deterministic from it), and every stored protected template becomes
  reversible to the extent partial-reconstruction attacks allow (see above).
  This makes `MASTER_SECRET` the single highest-value secret in a real
  deployment — `.env` is gitignored, `.env.example` ships with no real value,
  and there is no code path that logs or returns it.
- **Reconstruction risk.** Bounded by the transform's lossiness (many
  embedding dimensions collapse to fewer output bits) *given the template
  alone*; bounded much more weakly if the key material also leaks (see
  `docs/TEMPLATE_PROTECTION.md`). No quantitative reconstruction bound is
  claimed — only the qualitative direction (lossy without the key, weaker
  with it).
- **Cross-application linkability.** Deliberately broken by design: the same
  user's template differs per `application_id` (Experiment 3,
  `evaluation/privacy_metrics.py::experiment_diversity`), so a template
  observed in one application cannot be matched against the same user's
  template in another application, even by an entity that operates both.
