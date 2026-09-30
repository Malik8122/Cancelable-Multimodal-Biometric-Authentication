"""Centroid embedding for multi-pose enrollment.

One person looks different in every capture (pose, expression, lighting, glasses). Instead of enrolling whichever single
capture happened to be best, enrollment averages the embeddings of several guided poses and re-normalizes:

    centroid = normalize(mean(e_1 ... e_k))          (each e_i a unit-length 512-d FaceNet embedding)

The centroid sits in the middle of the person's cloud of embeddings, so a later single live capture - whatever the
small appearance change - lands closer to it than to any one pose. It is a plain mean + L2 normalization: no model is
retrained, and the result is fed through the unchanged BioHash / HKDF template generation.
"""

from __future__ import annotations

import numpy as np


def centroid_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """Average `embeddings` (each L2-normalized first) and L2-normalize the mean. Needs at least one embedding."""
    if not embeddings:
        raise ValueError("centroid_embedding needs at least one embedding.")
    stacked = np.stack([np.asarray(e, dtype=np.float64).ravel() for e in embeddings])
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot average a zero-length embedding.")
    mean = (stacked / norms).mean(axis=0)
    length = np.linalg.norm(mean)
    if length == 0:
        raise ValueError("The embeddings cancel out; no centroid direction exists.")
    return (mean / length).astype(np.float32)
