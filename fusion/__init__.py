"""Multimodal score fusion - see score_fusion.py::fuse_scores.

Contains no biometric inference logic of its own; it only combines
already-computed per-modality authentication scores produced by
`backend/services/*_service.py`.
"""

from __future__ import annotations
