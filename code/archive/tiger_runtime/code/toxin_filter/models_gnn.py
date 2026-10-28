"""TIGER-single encoder with a binary toxicity classification head.

Reuses the full ``PeptideEncoder`` from ``code/models.py`` (GNN graph branch +
sequence branch + 10-D global branch, ``SimpleSelfAttention`` fusion, gated by
``feature_modalities``) and attaches the same MLP head shape as
``SingleDeltaModel`` but emitting a single **logit** for BCE classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from code.models import PeptideEncoder
from code.utils.config import Config


class TIGERSingleToxin(nn.Module):
    """PeptideEncoder -> MLP head -> logit (1 = toxin, HC50 <= threshold)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = PeptideEncoder(cfg)
        self.head = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.emb_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(cfg.dropout_ratio),
            nn.Linear(cfg.emb_dim, 1),
        )

    def logits(self, batch) -> torch.Tensor:
        """[B] toxicity logits over a PyG Batch built by ``make_batch``."""
        return self.head(self.encoder(batch)).squeeze(-1)

    def forward(self, batch) -> torch.Tensor:  # keep module-callable like the DL baselines
        return self.logits(batch)
