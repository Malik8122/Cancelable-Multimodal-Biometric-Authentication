"""Offline tests for models/voice/export.py.

Regression coverage for a real bug found while building this: zipping the
output directory with `shutil.make_archive` right after `torch.load`-ing a
`.pt` file in that same directory reproducibly hung indefinitely on this
project's Windows dev environment (see export.py's `compress` branch for the
fix - explicit `zipfile` writes instead). These tests run with a short
`pytest-timeout`-independent guard by simply asserting completion; if this
regresses, the test process itself will hang, which is caught by CI/dev
timeouts rather than a clean assertion failure - an accepted tradeoff for
testing "does this hang" without adding a new test dependency.
"""

from __future__ import annotations

import zipfile

import torch
import torch.nn as nn

from models.voice.config import VoiceConfig
from models.voice.export import export_checkpoint


class _TinyVoiceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(80, 192)

    def forward(self, x):
        return self.fc(x)


def test_export_checkpoint_writes_pt_h5_and_config(tmp_path):
    checkpoint_path = tmp_path / "raw.pt"
    torch.save(_TinyVoiceNet().state_dict(), checkpoint_path)
    output_dir = tmp_path / "saved"

    produced = export_checkpoint(checkpoint_path, output_dir, config=VoiceConfig())

    assert produced["pt"].exists()
    assert produced["h5"].exists()
    assert produced["config"].exists()
    assert "archive" not in produced


def test_export_checkpoint_with_compress_produces_a_valid_zip_containing_all_three_files(tmp_path):
    checkpoint_path = tmp_path / "raw.pt"
    torch.save(_TinyVoiceNet().state_dict(), checkpoint_path)
    output_dir = tmp_path / "saved"

    produced = export_checkpoint(checkpoint_path, output_dir, config=VoiceConfig(), compress=True)

    assert produced["archive"].exists()
    with zipfile.ZipFile(produced["archive"]) as zip_file:
        assert zip_file.testzip() is None  # None means every member's checksum verified OK
        names = set(zip_file.namelist())
    assert names == {"voice_embedder.pt", "voice_embedder.h5", "training_config.json"}


def test_export_checkpoint_leaves_the_original_checkpoint_path_untouched_when_different_from_canonical(tmp_path):
    checkpoint_path = tmp_path / "raw.pt"
    torch.save(_TinyVoiceNet().state_dict(), checkpoint_path)
    output_dir = tmp_path / "saved"

    export_checkpoint(checkpoint_path, output_dir)

    assert checkpoint_path.exists()  # the source file wasn't moved, only copied
