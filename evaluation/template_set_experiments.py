"""Template-set experiments (docs/MULTI_TEMPLATE_ARCHITECTURE.md).

Same style as `evaluation/privacy_metrics.py`: plain functions returning plain
dicts, written with `privacy_metrics.save_csv_report`. The template-set
lifecycle runs on the shipped code (`backend.database.crud` +
`backend.services.base_service.ModalityService`) against a throw-away in-memory
SQLite database; only the embedding models are replaced by stubs that return
the embedding they are handed. Each simulated user is enrolled in BOTH
modalities (face 512-d, voice 192-d), so every template set
holds two templates.

- `experiment_template_set_diversity` -> `template_set_diversity.csv`:
  the same modality's templates in different sets (one embedding, different
  HKDF keys) must look unrelated (Hamming similarity ~0.5).
- `experiment_template_set_revocation` -> `template_set_revocation.csv`:
  authenticate before / after revoking the ACTIVE set; the revoked set's
  templates must not match; both modalities switch sets together.
- `experiment_template_set_promotion` -> `template_set_promotion.csv`:
  walk through set 1..N by repeated revocation; the genuine user keeps
  authenticating in every set, with all modalities in the same set every time.
- `experiment_template_set_exhaustion` -> `template_set_exhaustion.csv`:
  revoke until no STANDBY set is left; the next revocation is refused (HTTP 409).

The biometric-authorization gate on the HTTP endpoints (403 without a matching
capture) is covered by `tests/test_template_sets.py`, not measured here.

Run all four and write the CSVs:  python -m evaluation.template_set_experiments
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from evaluation.privacy_metrics import save_csv_report
from template_protection.biohash import generate_template
from template_protection.hkdf_keys import derive_key
from template_protection.matcher import compare
from template_protection.utils import l2_normalize, unpack_bits

RESULTS_DIR = Path(__file__).resolve().parent / "results"
_APPLICATION_ID = "experiment"
_MODALITY_DIMS = {"face": 512, "voice": 192}
#: Std-dev of the Gaussian added to an embedding to simulate a genuine re-capture
#: (~0.1 L2 norm on a unit vector, i.e. cosine ~0.995).
_RECAPTURE_NOISE = 0.004
_SECRET = "experiment-secret"


def _embedding(rng: np.random.Generator, dim: int) -> np.ndarray:
    return l2_normalize(rng.standard_normal(dim).astype(np.float32))


def _recapture(rng: np.random.Generator, embedding: np.ndarray) -> np.ndarray:
    return l2_normalize(embedding + _RECAPTURE_NOISE * rng.standard_normal(embedding.shape[0]).astype(np.float32))


def experiment_template_set_diversity(num_users: int = 20, pool_size: int = 4, output_bits: int = 256, seed: int = 0) -> dict:
    """Pairwise Hamming similarity between the same modality's templates in different sets."""
    rng = np.random.default_rng(seed)
    cross_set: dict[str, list[float]] = {m: [] for m in _MODALITY_DIMS}
    same_key_genuine: list[float] = []
    for user in range(num_users):
        for modality, dim in _MODALITY_DIMS.items():
            embedding = _embedding(rng, dim)
            keys = [
                derive_key(_SECRET, application_id=_APPLICATION_ID, user_id=f"u{user}", modality=modality, key_version=v)
                for v in range(1, pool_size + 1)
            ]
            templates = [generate_template(embedding, key, output_bits=output_bits) for key in keys]
            cross_set[modality] += [compare(templates[a], templates[b]) for a, b in itertools.combinations(range(pool_size), 2)]
            same_key_genuine.append(compare(templates[0], generate_template(_recapture(rng, embedding), keys[0], output_bits=output_bits)))
    everything = [value for values in cross_set.values() for value in values]
    report = {
        "experiment": "template_set_diversity",
        "modalities_per_set": len(_MODALITY_DIMS),
        "sets_per_user": pool_size,
        "num_users": num_users,
        "output_bits": output_bits,
        "cross_set_pairs": len(everything),
        "cross_set_similarity_mean": float(np.mean(everything)),
        "cross_set_similarity_std": float(np.std(everything)),
        "cross_set_similarity_min": float(np.min(everything)),
        "cross_set_similarity_max": float(np.max(everything)),
        "same_key_genuine_mean": float(np.mean(same_key_genuine)),
    }
    for modality, values in cross_set.items():
        report[f"cross_set_similarity_mean_{modality}"] = float(np.mean(values))
    return report


class _Harness:
    """A user enrolled in both modalities over an in-memory DB, via the real ModalityService."""

    def __init__(self, pool_size: int = 4, user_id: str = "u1"):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        from backend.config import Settings
        from backend.database.models import Base
        from backend.services.base_service import ModalityService

        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()
        self.settings = Settings(master_secret=_SECRET, template_pool_size=pool_size)
        self.user_id = user_id

        class _Pipeline:
            is_mock = False

            def embed(self, embedding):
                return embedding

        self.services = {m: ModalityService(m, _Pipeline(), self.settings) for m in _MODALITY_DIMS}
        for service in self.services.values():
            service.capture_gate = False  # stub input is an embedding/seed, not audio: no capture gate

    def enroll(self, embeddings: dict[str, np.ndarray]) -> None:
        for modality, embedding in embeddings.items():
            self.services[modality].enroll(self.db, embedding, self.user_id, _APPLICATION_ID)

    def authenticate(self, embeddings: dict[str, np.ndarray]) -> dict:
        return {m: self.services[m].authenticate(self.db, e, self.user_id, _APPLICATION_ID) for m, e in embeddings.items()}

    def revoke(self) -> tuple[int, int]:
        from backend.database import crud

        return crud.revoke_active_set_and_promote(self.db, self.user_id, _APPLICATION_ID)

    def rows(self):
        from backend.database import crud

        return crud.get_set_rows(self.db, self.user_id, _APPLICATION_ID)


def _enrolled_user(rng: np.random.Generator, pool_size: int = 4):
    harness = _Harness(pool_size)
    embeddings = {m: _embedding(rng, dim) for m, dim in _MODALITY_DIMS.items()}
    harness.enroll(embeddings)
    return harness, embeddings


def experiment_template_set_revocation(num_users: int = 20, seed: int = 1) -> dict:
    """Authenticate before / after revoking the ACTIVE set (genuine = small perturbation).

    `revoked_vs_new_similarity`: the new candidate (under the promoted set's key)
    compared with the REVOKED set's stored bits - ~0.5 means the revoked set no
    longer matches. `sets_switched_together` counts users whose modalities
    all reported the new set version.
    """
    rng = np.random.default_rng(seed)
    before, after, revoked_vs_new = [], [], []
    switched_together = 0
    for _ in range(num_users):
        harness, embeddings = _enrolled_user(rng)
        old_bits = {
            r.modality: unpack_bits(r.protected_template, num_bits=r.output_bits) for r in harness.rows() if r.template_set_version == 1
        }
        genuine = {m: _recapture(rng, e) for m, e in embeddings.items()}
        before += [r.hamming_similarity for r in harness.authenticate(genuine).values()]
        harness.revoke()
        results = harness.authenticate(genuine)
        after += [r.hamming_similarity for r in results.values()]
        switched_together += int({r.template_set_version for r in results.values()} == {2})
        for modality, result in results.items():
            key = derive_key(_SECRET, application_id=_APPLICATION_ID, user_id="u1", modality=modality, key_version=result.key_version)
            revoked_vs_new.append(compare(generate_template(genuine[modality], key, output_bits=256), old_bits[modality]))
    return {
        "experiment": "template_set_revocation",
        "num_users": num_users,
        "modalities_per_set": len(_MODALITY_DIMS),
        "genuine_similarity_before_mean": float(np.mean(before)),
        "genuine_similarity_after_mean": float(np.mean(after)),
        "revoked_vs_new_similarity_mean": float(np.mean(revoked_vs_new)),
        "users_with_all_modalities_in_new_set": switched_together,
    }


def experiment_template_set_promotion(num_users: int = 20, seed: int = 2) -> dict:
    """Promote STANDBY sets one after another; the genuine user must keep authenticating, in one set at a time."""
    rng = np.random.default_rng(seed)
    accepted = attempts = mixed_set_attempts = 0
    by_set: dict[int, list[float]] = {}
    for _ in range(num_users):
        harness, embeddings = _enrolled_user(rng)
        for _step in range(harness.settings.template_pool_size):
            results = harness.authenticate({m: _recapture(rng, e) for m, e in embeddings.items()})
            attempts += 1
            accepted += int(all(r.authenticated for r in results.values()))
            versions = {r.template_set_version for r in results.values()}
            mixed_set_attempts += int(len(versions) != 1)
            for r in results.values():
                by_set.setdefault(r.template_set_version, []).append(r.hamming_similarity)
            try:
                harness.revoke()
            except Exception:  # noqa: BLE001 - pool exhausted after the last set
                break
    report = {
        "experiment": "template_set_promotion",
        "num_users": num_users,
        "attempts": attempts,
        "accepted": accepted,
        "acceptance_rate": accepted / attempts,
        "attempts_mixing_sets": mixed_set_attempts,
    }
    for version, scores in sorted(by_set.items()):
        report[f"mean_similarity_active_set_{version}"] = float(np.mean(scores))
    return report


def experiment_template_set_exhaustion(pool_size: int = 4, seed: int = 3) -> dict:
    """Revoke until no STANDBY set is left; the next revocation is refused (HTTP 409)."""
    from backend.database import crud

    rng = np.random.default_rng(seed)
    harness, embeddings = _enrolled_user(rng, pool_size)
    successful = 0
    refused = False
    for _ in range(pool_size + 2):
        try:
            harness.revoke()
            successful += 1
        except crud.TemplatePoolExhaustedError as error:
            refused = str(error) == "Template set pool exhausted. Re-enrollment required."
            break
    results = harness.authenticate(embeddings)
    return {
        "experiment": "template_set_exhaustion",
        "pool_size": pool_size,
        "successful_revocations": successful,
        "expected_successful_revocations": pool_size - 1,
        "exhaustion_refused_http_409": refused,
        "active_set_still_authenticates": all(r.authenticated for r in results.values()),
        "final_active_set_version": next(iter(results.values())).template_set_version,
    }


def run_all(results_dir: Path = RESULTS_DIR) -> dict[str, dict]:
    reports = {
        "template_set_diversity.csv": experiment_template_set_diversity(),
        "template_set_revocation.csv": experiment_template_set_revocation(),
        "template_set_promotion.csv": experiment_template_set_promotion(),
        "template_set_exhaustion.csv": experiment_template_set_exhaustion(),
    }
    for name, report in reports.items():
        save_csv_report(report, results_dir / name)
    return reports


if __name__ == "__main__":
    for file_name, result in run_all().items():
        print(file_name)
        for key, value in result.items():
            print(f"  {key}: {value}")
