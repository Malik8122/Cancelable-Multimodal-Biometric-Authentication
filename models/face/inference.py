"""Face embedding model: InceptionResnetV1 pretrained on VGGFace2.

Produces a 512-dimensional embedding from an already-aligned 160x160 face
crop (see preprocessing/face.py). The pretrained VGGFace2 weights are strong
enough to use as a frozen feature extractor; models/face/saved/ holds any
fine-tuned checkpoint produced by notebooks/01_face_training_and_testing.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from models.common.base_embedder import BaseEmbedder

FACE_EMBEDDING_DIM = 512


class FaceEmbedder(BaseEmbedder):
    embedding_dim = FACE_EMBEDDING_DIM

    def __init__(self, checkpoint_path: str | Path | None = None, mock_mode: bool = False, device: str = "cpu"):
        self.device = device
        self._model = None
        super().__init__(checkpoint_path=checkpoint_path, mock_mode=mock_mode)

    def _build_model(self):
        from facenet_pytorch import InceptionResnetV1

        # `_build_model` is only ever called from `_load_checkpoint` below,
        # which immediately overwrites every weight via `load_state_dict` -
        # so materializing the real VGGFace2-pretrained weights first (the
        # default `pretrained="vggface2"`) is pure waste on this path: a
        # real, measured contributor to peak memory during construction, and
        # a real network download that never survives a restart on a
        # platform with no persistent disk (see
        # docs/AUTHENTICATION_RELIABILITY_REPORT.md's Render OOM
        # investigation).
        #
        # Plain `pretrained=None` is NOT enough on its own: facenet-pytorch
        # only creates the `logits` submodule (an `nn.Linear(512, 8631)`
        # classification head) in the SAME branch that downloads/loads the
        # real pretrained weights (`if pretrained is not None: self.logits =
        # ...; load_weights(...)`), so the two can't be separated through
        # that one argument. The project's checkpoint was saved from a model
        # that *does* have this submodule (its state_dict includes
        # `logits.weight`/`logits.bias`), so building without it makes
        # `load_state_dict` fail with "Unexpected key(s): logits.weight,
        # logits.bias" - confirmed by actually running this against the real
        # checkpoint before settling on the fix below.
        #
        # `classify=True, num_classes=8631` creates that same `logits` shape
        # through a *different*, independent branch in facenet-pytorch's
        # `__init__` (`if self.classify and self.num_classes is not None:
        # self.logits = ...`) that never touches the pretrained-weight
        # download - giving the checkpoint somewhere to load its saved
        # logits weights into, without ever materializing VGGFace2's real
        # values. `classify` is flipped back to `False` immediately after
        # construction, before this model is used for anything: `forward()`
        # returns raw classification logits when `classify=True` and the
        # L2-normalized embedding (what every caller here actually expects)
        # when `False`. Verified directly against the real checkpoint: this
        # produces bit-for-bit identical output (max abs diff 0.0) to the
        # original `pretrained="vggface2"` construction path.
        model = InceptionResnetV1(pretrained=None, classify=True, num_classes=8631)
        model.classify = False
        return model.eval().to(self.device)

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        import torch

        self._model = self._build_model()
        # `mmap=True` avoids fully materializing the checkpoint file into a
        # separate anonymous allocation before applying it; `assign=True`
        # replaces this model's (randomly-initialized) parameter tensors
        # directly with the loaded ones instead of copying values into the
        # already-allocated originals, letting those originals be freed
        # immediately rather than staying resident alongside the newly
        # loaded state_dict. Measured directly against this real checkpoint
        # (see docs/AUTHENTICATION_RELIABILITY_REPORT.md's Render OOM
        # investigation): this turns the loading step's memory delta from
        # +109 MB into a net -28 MB, with bit-for-bit identical output.
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
        )
        self._model.load_state_dict(state_dict, assign=True)
        self._model.eval()

    def _extract_embedding_impl(self, image: np.ndarray) -> np.ndarray:
        import torch

        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        tensor = (tensor - 127.5) / 128.0  # match facenet-pytorch's expected normalization
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self._model(tensor)
        return embedding.squeeze(0).cpu().numpy()
