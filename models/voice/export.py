"""Checkpoint export: `.pt` + `.h5` + `training_config.json`, optionally zipped.

`models/voice/train.py` already calls this logic inline at the end of a
training run - this module exists as its own file (per the spec) so the same
export step can also be re-run standalone (e.g. re-exporting `.h5` +
`training_config.json` for a `.pt` checkpoint that was produced some other
way), without re-running training. Reuses
`models/common/checkpoint_io.py::save_state_dict_as_h5` rather than
reimplementing HDF5 export - the same utility Face already
uses.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from models.common.checkpoint_io import save_state_dict_as_h5
from models.voice.config import VoiceConfig


def export_checkpoint(
    pt_checkpoint_path: str | Path,
    output_dir: str | Path,
    config: VoiceConfig | None = None,
    compress: bool = False,
) -> dict[str, Path]:
    """Given an existing `.pt` state_dict checkpoint, produce the full export bundle.

    Returns a dict of the produced paths (`pt`, `h5`, `config`, and `archive`
    if `compress=True`) so callers (this module's CLI, or a notebook cell)
    can report exactly what was written.
    """
    import torch

    pt_checkpoint_path = Path(pt_checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or VoiceConfig()

    canonical_pt_path = output_dir / "voice_embedder.pt"
    if pt_checkpoint_path != canonical_pt_path:
        shutil.copy2(pt_checkpoint_path, canonical_pt_path)

    state_dict = torch.load(canonical_pt_path, map_location="cpu")
    h5_path = output_dir / "voice_embedder.h5"
    save_state_dict_as_h5(state_dict, h5_path)
    del state_dict  # see note below - dropped before zipping regardless

    config_path = output_dir / "training_config.json"
    config.to_json(config_path)

    produced = {"pt": canonical_pt_path, "h5": h5_path, "config": config_path}

    if compress:
        # Deliberately `zipfile` with explicit per-file writes, not
        # `shutil.make_archive` (which walks + zips the whole directory):
        # on this project's Windows dev environment, calling
        # `shutil.make_archive` on a directory containing a `.pt` file that
        # was just `torch.load`-ed in the same process reproducibly hangs
        # indefinitely (observed, not theoretical - see the git history for
        # this file). Writing each already-known file explicitly sidesteps
        # whatever directory-walk/file-handle interaction causes that.
        archive_path = output_dir / "voice_embedder_export.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for produced_file in (canonical_pt_path, h5_path, config_path):
                zip_file.write(produced_file, arcname=produced_file.name)
        produced["archive"] = archive_path

    return produced


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a Voice checkpoint to the standard .pt/.h5/config bundle.")
    parser.add_argument("--checkpoint", required=True, help="Path to an existing voice_embedder .pt file.")
    parser.add_argument("--output-dir", default="models/voice/saved")
    parser.add_argument("--compress", action="store_true", help="Also write a .zip archive of the output directory.")
    args = parser.parse_args()

    produced_paths = export_checkpoint(args.checkpoint, args.output_dir, compress=args.compress)
    for label, path in produced_paths.items():
        print(f"{label}: {path}")
