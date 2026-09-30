"""Round-trip test for the .h5 checkpoint export/import utility.

Uses a tiny real nn.Module (not one of the full biometric backbones) so this
runs fast and offline - it only needs to prove save_state_dict_as_h5 /
load_h5_as_state_dict preserve a state_dict exactly, independent of which
model architecture produced it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.common.checkpoint_io import load_h5_as_state_dict, save_state_dict_as_h5


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_h5_round_trip_preserves_all_keys_and_values(tmp_path):
    model = _TinyNet()
    original_state_dict = model.state_dict()

    h5_path = tmp_path / "tiny_net.h5"
    save_state_dict_as_h5(original_state_dict, h5_path)

    assert h5_path.exists()

    restored_state_dict = load_h5_as_state_dict(h5_path)

    assert set(restored_state_dict.keys()) == set(original_state_dict.keys())
    for key, original_tensor in original_state_dict.items():
        torch.testing.assert_close(restored_state_dict[key], original_tensor)


def test_h5_round_trip_loads_into_a_fresh_model(tmp_path):
    model = _TinyNet()
    h5_path = tmp_path / "tiny_net.h5"
    save_state_dict_as_h5(model.state_dict(), h5_path)

    fresh_model = _TinyNet()
    fresh_model.load_state_dict(load_h5_as_state_dict(h5_path))

    sample_input = torch.randn(1, 8)
    torch.testing.assert_close(model(sample_input), fresh_model(sample_input))


def test_h5_export_creates_parent_directories(tmp_path):
    nested_path = tmp_path / "does" / "not" / "exist" / "model.h5"
    save_state_dict_as_h5(_TinyNet().state_dict(), nested_path)
    assert nested_path.exists()
