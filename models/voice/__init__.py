"""Voice (speaker verification) embedding model.

Mirrors models/face/'s shape: preprocessing
lives in preprocessing/voice.py, this package owns the model architecture,
training, and inference. See docs/VOICE_MODEL.md for the full design.
"""

from __future__ import annotations
