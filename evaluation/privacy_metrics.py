"""Phase 2 privacy/template-protection experiments.

Mirrors `evaluation/experiments.py`'s style (plain functions over numpy
arrays, returning a plain `dict`) but evaluates `template_protection/`
instead of raw embeddings:

- Experiment 1 (`experiment_similarity_preservation`): does protected-template
  similarity track raw-embedding similarity closely enough to still be usable
  for recognition?
- Experiment 2 (`experiment_revocability`): rotating a user's key must make
  the new template unlinkable to the old one.
- Experiment 3 (`experiment_diversity`): the same biometric enrolled under
  different application IDs must produce unlinkable templates.
- Experiment 4 (`experiment_protected_far_frr`): FAR/FRR/EER computed on
  protected-template Hamming scores instead of raw cosine similarity, so it
  can be compared directly against Experiment 1's (Phase 1) unprotected
  numbers.

`save_csv_report` writes any of these dicts to `evaluation/results/` (already
`.gitignore`d for `*.csv`, with `.gitkeep` keeping the directory tracked).
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np

from evaluation.metrics import accuracy_at_threshold, compute_eer, compute_far_frr, cosine_similarity
from template_protection.biohash import DEFAULT_OUTPUT_BITS, generate_template
from template_protection.hkdf_keys import KeyMaterial, derive_key
from template_protection.matcher import compare


def experiment_similarity_preservation(
    embeddings: list[np.ndarray],
    labels: list[str],
    master_secret: bytes | str,
    application_id: str,
    modality: str,
    output_bits: int = DEFAULT_OUTPUT_BITS,
) -> dict:
    """Experiment 1: raw cosine similarity vs. protected Hamming similarity.

    Every embedding is treated as belonging to its own enrolled user
    (`labels[i]` -> `derive_key(user_id=labels[i], ...)`), matching how
    `backend/services/*.py` derives one key per user+modality+application.
    Reports the Pearson correlation between the two similarity series across
    every pair - a high correlation means the cancelable transform preserves
    enough recognition-relevant structure to still be usable (the spec's
    "Recognition Preservation" principle), not just that it's "protected."
    """
    raw_scores: list[float] = []
    protected_scores: list[float] = []

    keys = {label: derive_key(master_secret, application_id=application_id, user_id=label, modality=modality) for label in set(labels)}
    templates = [generate_template(embedding, keys[label], output_bits=output_bits) for embedding, label in zip(embeddings, labels)]

    for (emb_a, template_a), (emb_b, template_b) in itertools.combinations(zip(embeddings, templates), 2):
        raw_scores.append(cosine_similarity(emb_a, emb_b))
        protected_scores.append(compare(template_a, template_b, metric="hamming"))

    raw_array, protected_array = np.array(raw_scores), np.array(protected_scores)
    correlation = float(np.corrcoef(raw_array, protected_array)[0, 1]) if len(raw_array) > 1 else float("nan")

    return {
        "experiment": "similarity_preservation",
        "modality": modality,
        "num_pairs": len(raw_scores),
        "output_bits": output_bits,
        "raw_cosine_scores": raw_array,
        "protected_hamming_scores": protected_array,
        "correlation": correlation,
    }


def experiment_revocability(
    embedding: np.ndarray,
    master_secret: bytes | str,
    application_id: str,
    user_id: str,
    modality: str,
    output_bits: int = DEFAULT_OUTPUT_BITS,
    num_rotations: int = 5,
) -> dict:
    """Experiment 2: rotate key_version 1..num_rotations, report pairwise Hamming distances.

    A working revocation scheme should put every pairwise distance near 0.5
    (statistically unrelated bit strings) - values close to 0 would mean
    rotation isn't actually changing the template.
    """
    templates = [
        generate_template(
            embedding,
            derive_key(master_secret, application_id=application_id, user_id=user_id, modality=modality, key_version=version),
            output_bits=output_bits,
        )
        for version in range(1, num_rotations + 1)
    ]

    pairwise_hamming_distances = [
        float(np.mean(template_a != template_b)) for template_a, template_b in itertools.combinations(templates, 2)
    ]

    return {
        "experiment": "revocability",
        "modality": modality,
        "user_id": user_id,
        "output_bits": output_bits,
        "num_rotations": num_rotations,
        "pairwise_hamming_distances": pairwise_hamming_distances,
        "mean_hamming_distance": float(np.mean(pairwise_hamming_distances)),
    }


def experiment_diversity(
    embedding: np.ndarray,
    master_secret: bytes | str,
    user_id: str,
    modality: str,
    application_ids: list[str],
    output_bits: int = DEFAULT_OUTPUT_BITS,
) -> dict:
    """Experiment 3: same embedding + user, different application IDs -> unlinkable templates.

    Same expected result as `experiment_revocability` (pairwise distances near
    0.5), but varying `application_id` at a fixed `key_version` instead of
    varying `key_version` - this is what stops a single compromised template
    at one application from being usable to impersonate the same person at
    another application.
    """
    templates = {
        application_id: generate_template(
            embedding,
            derive_key(master_secret, application_id=application_id, user_id=user_id, modality=modality),
            output_bits=output_bits,
        )
        for application_id in application_ids
    }

    pairwise_hamming_distances = {
        f"{app_a}_vs_{app_b}": float(np.mean(templates[app_a] != templates[app_b]))
        for app_a, app_b in itertools.combinations(application_ids, 2)
    }

    return {
        "experiment": "diversity",
        "modality": modality,
        "user_id": user_id,
        "output_bits": output_bits,
        "application_ids": application_ids,
        "pairwise_hamming_distances": pairwise_hamming_distances,
        "mean_hamming_distance": float(np.mean(list(pairwise_hamming_distances.values()))),
    }


def experiment_protected_far_frr(
    embeddings: list[np.ndarray],
    labels: list[str],
    master_secret: bytes | str,
    application_id: str,
    modality: str,
    output_bits: int = DEFAULT_OUTPUT_BITS,
) -> dict:
    """Experiment 4: FAR/FRR/EER computed on protected-template Hamming scores.

    Structurally the protected-template counterpart of
    `evaluation/experiments.py::run_modality_experiment`, so the two can be
    reported side by side (unprotected EER vs. protected EER) as the spec's
    "false match statistics using protected templates" deliverable.
    """
    keys = {label: derive_key(master_secret, application_id=application_id, user_id=label, modality=modality) for label in set(labels)}
    templates = [generate_template(embedding, keys[label], output_bits=output_bits) for embedding, label in zip(embeddings, labels)]

    genuine, impostor = [], []
    for (template_a, label_a), (template_b, label_b) in itertools.combinations(zip(templates, labels), 2):
        score = compare(template_a, template_b, metric="hamming")
        (genuine if label_a == label_b else impostor).append(score)

    genuine_scores, impostor_scores = np.array(genuine), np.array(impostor)
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        raise ValueError(
            "Need at least one genuine pair and one impostor pair; make sure the "
            "sample set has multiple images per identity and multiple identities."
        )

    eer, eer_threshold = compute_eer(genuine_scores, impostor_scores)
    accuracy = accuracy_at_threshold(genuine_scores, impostor_scores, eer_threshold)
    far, frr = compute_far_frr(genuine_scores, impostor_scores, eer_threshold)

    return {
        "experiment": "protected_far_frr",
        "modality": modality,
        "output_bits": output_bits,
        "num_genuine_pairs": len(genuine_scores),
        "num_impostor_pairs": len(impostor_scores),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "far_at_eer_threshold": far,
        "frr_at_eer_threshold": frr,
        "accuracy_at_eer_threshold": accuracy,
    }


def save_csv_report(report: dict, path: str | Path) -> None:
    """Flatten a report dict to a two-column (key, value) CSV under `evaluation/results/`.

    Array/list-valued fields are written as a single semicolon-joined cell
    rather than exploded into rows, keeping every report - regardless of
    which `experiment_*` function produced it - representable in the same
    simple format.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["field", "value"])
        for key, value in report.items():
            if isinstance(value, (np.ndarray, list)):
                value = ";".join(str(item) for item in np.asarray(value).tolist())
            elif isinstance(value, dict):
                value = ";".join(f"{sub_key}={sub_value}" for sub_key, sub_value in value.items())
            writer.writerow([key, value])
