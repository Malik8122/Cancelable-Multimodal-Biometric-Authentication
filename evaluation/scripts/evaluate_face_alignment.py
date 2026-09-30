"""Face alignment diagnostics and baseline-vs-aligned evaluation (entry point).

Delegates to evaluation/ieee/face_alignment_eval.py, which holds the protocol: examples (original + MTCNN box +
landmarks + baseline crop + aligned crop + canonical template, public LFW only), geometric quality before/after
alignment, identical-pair comparison of the systems A-F, threshold sweep, fusion, robustness and latency.

Run: python -m evaluation.scripts.evaluate_face_alignment [extract quality compare threshold fusion robustness latency examples]
"""

from __future__ import annotations

import sys

from evaluation.ieee.common import timed
from evaluation.ieee.face_alignment_eval import STEPS

if __name__ == "__main__":
    for step in sys.argv[1:] or list(STEPS):
        with timed(f"face_alignment_{step}"):
            STEPS[step]()
