"""HDF5 export/import for trained checkpoints.

The canonical, loadable checkpoint format for this project is PyTorch's
native `.pt` (`torch.save`/`torch.load` + `state_dict`) - that's what
`BaseEmbedder._load_checkpoint` in every models/*/inference.py uses, and
it's the natural fit for the InceptionResnetV1/ResNet backbones in use.

This module additionally exports the same trained weights to `.h5`
(HDF5) purely for interoperability and easy inspection outside PyTorch
(e.g. `h5py`, MATLAB, or a future non-PyTorch tool) - every training
notebook calls `save_state_dict_as_h5` right after `torch.save(...)`, so
both files always exist side by side in `models/<modality>/saved/`.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch


def save_state_dict_as_h5(state_dict: "OrderedDict[str, torch.Tensor]", path: str | Path) -> None:
    """Write a PyTorch state_dict to an HDF5 file, one dataset per parameter/buffer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5file:
        h5file.attrs["framework"] = "pytorch"
        h5file.attrs["format_version"] = 1
        h5file.attrs["note"] = (
            "Interoperability export. The canonical, loadable checkpoint for this "
            "project is the sibling .pt file - see models/common/base_embedder.py."
        )
        for key, tensor in state_dict.items():
            h5file.create_dataset(key, data=tensor.detach().cpu().numpy())


def load_h5_as_state_dict(path: str | Path) -> "OrderedDict[str, torch.Tensor]":
    """Read an HDF5 file written by `save_state_dict_as_h5` back into a state_dict.

    Provided for interoperability/verification (e.g. round-trip testing); the
    production inference path still loads the sibling `.pt` file directly.
    """
    state_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    with h5py.File(path, "r") as h5file:
        for key in h5file.keys():
            state_dict[key] = torch.from_numpy(np.array(h5file[key]))
    return state_dict
