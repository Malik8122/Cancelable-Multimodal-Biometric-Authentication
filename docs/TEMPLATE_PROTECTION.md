# Template Protection

This document explains `template_protection/`: the cancelable, BioHashing-style
transform that sits between a raw biometric embedding and what actually gets
stored in the database. See `docs/ARCHITECTURE.md` for where this fits in the
overall pipeline and `docs/PRIVACY_AND_SECURITY.md` for the broader
threat/privacy analysis this module feeds into.

## Why "cancelable"

A password can be changed after a breach. A face or a voice
cannot. Storing a raw embedding as a permanent credential means a database
compromise permanently compromises that person's biometric identity across
every system that trusted it. A **cancelable** template is derived from the
embedding through a keyed, revocable transform: if a template is ever
compromised, the system issues a new one from a new key without needing a new
biometric characteristic - the same face, under a different key, produces an
unrelated template.

## The pipeline, stage by stage

```text
embedding (unit vector, dim D)                    [from embeddings/pipelines.py]
     |
     v
KeyMaterial = derive_key(master_secret, application_id, user_id, modality, key_version)
     |  (hkdf_keys.py - HKDF-SHA256, three independent seeds)
     v
projection_matrix = build_orthonormal_projection(projection_seed, output_bits, D)
     |  (transform.py)
     v
projected = projection_matrix @ embedding          -> real values in [-1, 1]
     |
     v
quantized = quantize(projected, threshold_seed)    -> bits, keyed threshold
     |
     v
template = apply_permutation(quantized, permutation_seed)   -> the protected template
```

`biohash.py::generate_template` is the single function that runs this whole
pipeline; `hkdf_keys.py::derive_key` is what produces the three seeds it
needs. Nothing here ever touches the database or an HTTP request - both live
one layer up, in `backend/services/*.py`.

### 1. Key derivation (HKDF-SHA256)

`derive_key(master_secret, application_id, user_id, modality, key_version)`
returns three independent 32-byte seeds (`projection_seed`,
`permutation_seed`, `threshold_seed`), each from its own
`cryptography.hazmat.primitives.kdf.hkdf.HKDF` call, salted by a SHA-256 hash
of the four identifying fields and distinguished by a per-seed `info` string.

This is deterministic: the same five inputs always produce the same three
seeds, anywhere, in any process - required so that authenticating later
reproduces the exact template enrollment produced. Changing **any** input
changes all three seeds:

- Different `user_id` or `modality` -> unrelated key material for a
  different identity/modality (expected; these should never collide).
- Different `application_id`, same user/modality -> the **diversity**
  property: the same person enrolled in two different applications gets two
  unlinkable templates, so a template leaked from one application is useless
  against the other.
- Different `key_version`, everything else fixed -> the **revocation**
  property: rotating the key for a compromised template invalidates it
  without needing a new biometric sample.

`master_secret` is a caller-supplied argument (from `backend/config.py`'s
`MASTER_SECRET`, sourced from the environment) - never hardcoded, never
stored alongside a template.

### 2. Orthonormal random projection

`transform.build_orthonormal_projection(seed, output_bits, embedding_dim)`
builds an `(output_bits, embedding_dim)` matrix whose rows are unit vectors,
seeded deterministically from `projection_seed` via `numpy.random.default_rng`
(PCG64 - a fixed, portable algorithm, not system entropy) and orthonormalized
via QR decomposition of a Gaussian random matrix.

**Honest limitation:** true orthonormality across *every* row is only
possible when `output_bits <= embedding_dim` (a real matrix's rank can't
exceed its smaller dimension). Face embeddings are 512-dimensional and
voice 192 (see `models/*/inference.py`). When a caller
requests more output bits than the embedding has dimensions (e.g. a 256-bit
voice template, the deployed case), this function composes multiple independent orthonormal
*blocks* - each block is internally orthonormal, but rows in different blocks
are not orthogonal to each other. This is a deliberate, documented tradeoff:
the alternative would be silently degrading to some other, weaker scheme
without saying so. `template_protection.biohash.DEFAULT_OUTPUT_BITS = 128` is
chosen specifically so the default case never hits this limitation, for any
modality in this project.

Projecting the (unit-norm) embedding through a (unit-norm-row) matrix gives
values bounded in `[-1, 1]` by the Cauchy-Schwarz inequality - `quantize`
relies on this.

### 3. Keyed quantization

`transform.quantize(projected, threshold_seed)` binarizes each projected
value against a **per-position random threshold**, drawn from
`Normal(0, 0.5 * std(projected))` and seeded by `threshold_seed`.

Two design points worth calling out:

- **Centered at zero, not a fixed magic number.** An earlier version of this
  code used `Uniform(-1, 1)` thresholds. That's wrong: `projected` values for
  a 512-dimensional embedding are typically tiny (~1/sqrt(512) ≈ 0.044,
  since each component is a dot product of two unit vectors), so a
  fixed `[-1, 1]`-scale threshold overwhelms the actual signal almost every
  time, and quantization ends up depending on the threshold alone -
  collapsing *every* embedding to nearly the same template regardless of
  input. Scaling the threshold's spread to `projected`'s own standard
  deviation keeps it calibrated to whatever `output_bits`/`embedding_dim`
  combination is in play.
- **Keyed, not a plain sign function.** The jitter is small enough that
  `quantize`'s output is still primarily a sign-split of the (embedding-driven)
  projected value - which is what makes two different embeddings under the
  same key discriminable - while still shifting exactly which positions land
  on which side of that split depending on the key.

### 4. Permutation

`transform.apply_permutation(bits, permutation_seed)` reorders the bit vector
with a key-derived random permutation. A permutation alone adds no entropy
(each output bit is still exactly as guessable as some input bit), but
combined with the keyed projection and quantization above, it means two
templates derived under different keys share no positional structure at all -
this is what the revocability/diversity experiments in
`evaluation/privacy_metrics.py` measure.

## Revocation workflow (template sets)

Enrollment derives, from each modality's embedding, one template per **template set**, each
under its own key version. A template set is a complete multimodal credential and the unit of
revocation (full design: `docs/MULTI_TEMPLATE_ARCHITECTURE.md`):

```text
                 Face              Voice
Template Set 1   V1  ---HKDF v1--  V1  ---HKDF v1--   ACTIVE
Template Set 2   V2  ---HKDF v2--  V2  ---HKDF v2--   STANDBY
Template Set 3   V3                V3                 STANDBY
Template Set 4   V4                V4                 STANDBY
                 (embeddings discarded after generation)

revoke:  Set 1 ACTIVE -> REVOKED,  Set 2 STANDBY -> ACTIVE   (both modalities together)
```

Authentication regenerates each candidate under the ACTIVE set's key versions and compares it
with that set's template only; a runtime check refuses to authenticate if active rows ever come
from different sets. Because each key version yields a statistically unrelated template (~0.50
Hamming similarity), a revoked set is unlinkable to its replacement. With no STANDBY set left,
revocation returns HTTP 409 and the user must re-enroll (or generate a new set with a capture
that authenticates against the active one).

A template set contains templates only for the modalities enrolled at that moment. Enrolling another modality
later generates that modality's templates for the existing live sets under fresh key versions and leaves the other
modalities' stored templates untouched (byte for byte), so the set stays one credential.

Revocation and the other management actions need no login; they require **biometric
authorization** - a fresh capture that authenticates against the ACTIVE set (403 otherwise) - so a
stranger cannot exhaust another user's set pool.

The earlier design - `revoke.py::revoke_template` re-deriving at `new_key_version` from a *fresh
capture* - remains as a library function, but `POST /revoke-template` no longer re-keys anything: a
protected template is a lossy transform and cannot be re-keyed after the fact, so the replacement
sets are generated while the embeddings are available and stored as STANDBY.

`backend/database/crud.py::revoke_active_set_and_promote` performs the transition in one
transaction; partial unique indexes guarantee one ACTIVE row per modality and one live template per
modality per set.

## Diversity workflow

Identical mechanism, different lever: instead of bumping `key_version`, use a
different `application_id` for the same `user_id`/`modality`. This is what
prevents one application's compromised template from being replayable
against a different application the same person is also enrolled in -
`evaluation/privacy_metrics.py::experiment_diversity` measures the resulting
pairwise Hamming distances (expected: near 0.5, i.e. statistically
unrelated).

## Security assumptions and limitations

Stated plainly, per this project's "don't overclaim" policy (see
`docs/PRIVACY_AND_SECURITY.md`):

- **This is information-lossy, not cryptographically one-way.** Collapsing a
  256- or 512-dimensional real vector to 128 (or more) bits is a many-to-one
  map, which makes reconstructing the exact original embedding from the
  template *alone* infeasible. That is a data-transformation argument, **not**
  a computational-hardness proof like a cryptographic hash function has. No
  claim of formal irreversibility is made anywhere in this codebase.
- **Key compromise weakens the guarantee.** If an attacker recovers both a
  protected template *and* the key material that produced it (the three HKDF
  seeds), they know the exact projection matrix, thresholds, and permutation
  used - at that point, partial reconstruction of the underlying embedding is
  a studied attack in the BioHashing literature (this is exactly why
  `MASTER_SECRET` must be treated as the single most sensitive value in the
  deployment - see `backend/config.py` and `.env.example`).
- **Constant-time comparison is partial.** `template_protection.utils.constant_time_equals`
  (used for the exact-match fast path in `matcher.compare`) is genuinely
  constant-time for byte-equality. Hamming-distance computation itself is
  not meaningfully hardened against timing side channels beyond that fast
  path - this is disclosed rather than silently assumed away.
- **Recognition preservation is empirical, not guaranteed for every input
  distribution.** `evaluation/privacy_metrics.py::experiment_similarity_preservation`
  measures how well protected-template similarity tracks raw-embedding
  similarity for a given sample set; it is not a proof that this holds for
  every possible biometric population.
- **Mock-mode embeddings produce mock-mode templates.** When a modality's
  checkpoint is missing, templates generated from a mock
  embedding are not biometrically meaningful and must not be treated as real
  authentication material outside of testing/demos.
