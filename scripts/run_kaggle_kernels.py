"""Push all 3 Phase 1 Kaggle Kernels, wait for them to finish, and pull the
trained checkpoints back into models/<modality>/saved/.

Requires: `pip install kaggle` and a Kaggle API token at ~/.kaggle/kaggle.json
(see kaggle_kernels/README.md).

Usage:
    python scripts/run_kaggle_kernels.py                 # all 3 modalities
    python scripts/run_kaggle_kernels.py --only face      # just one
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_KERNELS_DIR = REPO_ROOT / "kaggle_kernels"
POLL_INTERVAL_SECONDS = 30
MAX_WAIT_SECONDS = 60 * 60  # Kaggle GPU kernels are capped at a few hours anyway


def find_kaggle_executable() -> str:
    """Resolve the kaggle CLI even when its install location isn't on PATH.

    pip installs the `kaggle` console script into the Python user Scripts
    directory, which isn't always on PATH (notably on Windows) - this checks
    PATH first, then falls back to `<user base>/Scripts/kaggle(.exe)`.
    """
    found = shutil.which("kaggle")
    if found:
        return found

    import os
    import sysconfig

    scripts_dir = Path(sysconfig.get_path("scripts", f"{os.name}_user"))
    for candidate in (scripts_dir / "kaggle.exe", scripts_dir / "kaggle"):
        if candidate.exists():
            return str(candidate)

    raise SystemExit(
        "Could not find the `kaggle` CLI executable. Run `pip install kaggle` "
        "and make sure it's on PATH, or adjust find_kaggle_executable()."
    )


KAGGLE_EXE = find_kaggle_executable()

MODALITIES = {
    "face": ("face_training", "face-embedding-training-phase-1"),
    "voice": ("voice_training", "voice-embedding-training-phase-1-extension"),
}


def get_kaggle_username() -> str:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise SystemExit(
            f"No Kaggle API token found at {kaggle_json}. "
            "See kaggle_kernels/README.md for how to create one."
        )
    with open(kaggle_json) as f:
        return json.load(f)["username"]


def patch_kernel_metadata(kernel_dir: Path, username: str) -> str:
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    slug = metadata["id"].split("/")[-1]
    metadata["id"] = f"{username}/{slug}"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata["id"]


def run(cmd: list[str]) -> str:
    import os

    # The kaggle CLI reads notebook files as UTF-8; on Windows the default
    # subprocess locale encoding (cp1252) can't decode the unicode arrows/
    # checkmarks used in these notebooks' markdown cells, so force UTF-8 mode
    # for the child process explicitly rather than stripping those characters.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    output = result.stdout + result.stderr
    print(output)
    # The kaggle CLI sometimes reports a failure as plain stdout text with
    # exit code 0 (e.g. "Kernel push error: ...") rather than a non-zero
    # return code - check for that explicitly rather than trusting the
    # return code alone.
    if result.returncode != 0 or "kernel push error" in output.lower():
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{output}")
    return output


def push_kernel(kernel_dir: Path) -> None:
    run([KAGGLE_EXE, "kernels", "push", "-p", str(kernel_dir)])


def wait_for_completion(kernel_id: str) -> None:
    waited = 0
    while waited < MAX_WAIT_SECONDS:
        status_output = run([KAGGLE_EXE, "kernels", "status", kernel_id]).lower()
        if "complete" in status_output:
            print(f"{kernel_id}: complete")
            return
        if "error" in status_output or "cancel" in status_output:
            # Catches CANCELLED as well as CANCEL_ACKNOWLEDGED/CANCEL_REQUESTED -
            # matching only the exact string "cancelled" missed those states.
            raise RuntimeError(f"{kernel_id} failed or was cancelled - status: {status_output.strip()}")
        print(f"{kernel_id}: still running, waiting {POLL_INTERVAL_SECONDS}s...")
        time.sleep(POLL_INTERVAL_SECONDS)
        waited += POLL_INTERVAL_SECONDS
    raise TimeoutError(f"{kernel_id} did not complete within {MAX_WAIT_SECONDS}s")


def pull_output_and_install_checkpoints(kernel_id: str, modality: str) -> None:
    output_dir = REPO_ROOT / "kaggle_output" / modality
    output_dir.mkdir(parents=True, exist_ok=True)
    run([KAGGLE_EXE, "kernels", "output", kernel_id, "-p", str(output_dir)])

    saved_source = output_dir / "repo" / "models" / modality / "saved"
    saved_dest = REPO_ROOT / "models" / modality / "saved"
    saved_dest.mkdir(parents=True, exist_ok=True)

    copied = []
    for pattern in ("*.pt", "*.h5"):
        for src_file in saved_source.glob(pattern):
            dest_file = saved_dest / src_file.name
            shutil.copy2(src_file, dest_file)
            copied.append(dest_file)

    if not copied:
        raise RuntimeError(
            f"No .pt/.h5 checkpoint found under {saved_source} - check the kernel's "
            "output for errors before assuming training succeeded."
        )
    for path in copied:
        print(f"Installed checkpoint: {path.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(MODALITIES), help="Run a single modality instead of all 3")
    args = parser.parse_args()

    username = get_kaggle_username()
    targets = [args.only] if args.only else list(MODALITIES)

    for modality in targets:
        subdir, slug = MODALITIES[modality]
        kernel_dir = KAGGLE_KERNELS_DIR / subdir
        print(f"\n=== {modality} ===")
        kernel_id = patch_kernel_metadata(kernel_dir, username)
        push_kernel(kernel_dir)
        wait_for_completion(kernel_id)
        pull_output_and_install_checkpoints(kernel_id, modality)

    print("\nDone. Review the changes, then:")
    print("  git add models/*/saved/*.pt models/*/saved/*.h5")
    print('  git commit -m "Add trained checkpoints from Kaggle GPU training"')
    print("  git push")


if __name__ == "__main__":
    main()
