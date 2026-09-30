"""Reproduce the whole IEEE evaluation package, in dependency order.

    python -m evaluation.ieee.run_all            # everything (several hours on a laptop CPU)
    python -m evaluation.ieee.run_all --skip-extract --skip-benchmarks

Prerequisites (see evaluation/IEEE_EVALUATION_PACKAGE.md, "Reproducibility"):
- `git lfs pull` (real checkpoints), Kaggle API credentials (voice dataset), internet (LFW).
- datasets under data/ (gitignored): data/lfw, data/voxceleb_subset.
Benchmarks (latency, memory) run last and alone so that no other workload distorts the timings.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from evaluation.ieee.common import REPO, timed

STEPS = [
    ("extract", [sys.executable, "-m", "evaluation.ieee.extract_embeddings"]),
    ("audit", [sys.executable, "-m", "evaluation.ieee.audit", "audit"]),
    ("verification", [sys.executable, "-m", "evaluation.ieee.verification"]),
    ("experiments", [sys.executable, "-m", "evaluation.ieee.experiments"]),
    ("robustness", [sys.executable, "-m", "evaluation.ieee.robustness"]),
    ("storage", [sys.executable, "-m", "evaluation.ieee.storage"]),
    ("pad", [sys.executable, "-m", "evaluation.ieee.pad_evaluation"]),
    ("figures", [sys.executable, "-m", "evaluation.ieee.figures"]),
    ("latency", [sys.executable, "-m", "scripts.benchmark_latency"]),
    ("memory", [sys.executable, "-m", "scripts.benchmark_memory"]),
    ("software", [sys.executable, "-m", "evaluation.ieee.audit", "software"]),
    ("reports", [sys.executable, "-m", "evaluation.ieee.reports"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-benchmarks", action="store_true")
    a = ap.parse_args()
    for name, cmd in STEPS:
        if (name == "extract" and a.skip_extract) or (name in ("latency", "memory") and a.skip_benchmarks):
            continue
        with timed(f"run_all:{name}"):
            subprocess.run(cmd, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
