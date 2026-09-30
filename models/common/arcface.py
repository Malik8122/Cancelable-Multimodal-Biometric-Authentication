"""ArcMarginProduct: additive angular margin loss head (ArcFace).

Shared fine-tuning head used by the face training notebook
(notebooks/01_face_*) and the voice training loop (models/voice/losses.py)
so every modality is fine-tuned the same, well-documented way (Deng et al., "ArcFace: Additive
Angular Margin Loss for Deep Face Recognition", CVPR 2019). Only used
during training - inference-time embedders (models/*/inference.py) never
import this, since at inference we only need the backbone's raw embedding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=0, max=1))
        phi = cosine * self.cos_m - sine * self.sin_m
        # numerically safer variant: fall back to plain cosine minus margin
        # when the angle would otherwise wrap past pi (Deng et al., eq. 6)
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s
