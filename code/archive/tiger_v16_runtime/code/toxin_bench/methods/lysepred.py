"""LysePred (multi-scale CNN, lincubator/LysePred) — ported to the unified harness.

Architecture and defaults follow the repo exactly:
  Embedding(26,128,pad=0) with their shuffled prot vocab ([PAD]=0,[CLS]=1,...)
  5 parallel Conv2d(1,128,(k,128)) branches, k in [2,4,8,16,32], + BatchNorm2d + ReLU
  global max-pool per branch -> concat 640 -> Dropout(0.2) -> Linear(640,128) -> Linear(128,2)
  Adam(lr=1e-4, wd=2.5e-3), batch 32, CE + loss flooding b=0.02,
  CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6), 100 epochs,
  no early stopping, unweighted CE, no class balancing.

Deviations (harness-imposed, recorded in META): torch seeded per fold for
reproducibility (upstream is unseeded); the final-epoch model is used instead
of upstream's best-test-epoch checkpoint (test-based selection is not valid
under our protocol); [UZOB]->X mapping; fixed max_len = max train length + 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# vocab from LysePred data/statistic/proteintoken2index.pkl
VOCAB = {"[PAD]": 0, "[CLS]": 1, "B": 2, "Q": 3, "I": 4, "D": 5, "M": 6, "V": 7, "G": 8,
         "K": 9, "Y": 10, "P": 11, "H": 12, "Z": 13, "W": 14, "U": 15, "A": 16, "N": 17,
         "F": 18, "R": 19, "S": 20, "C": 21, "E": 22, "L": 23, "T": 24, "X": 25}
import re  # noqa: E402


def encode(seqs: list[str], max_len: int) -> np.ndarray:
    rows = []
    for s in seqs:
        s = re.sub(r"[UZOB]", "X", s.upper())
        ids = [VOCAB["[CLS]"]] + [VOCAB.get(ch, VOCAB["X"]) for ch in s]
        ids = ids[:max_len] + [VOCAB["[PAD]"]] * max(0, max_len - len(ids))
        rows.append(ids)
    return np.asarray(rows, dtype=np.int64)


class Net(nn.Module):
    def __init__(self, vocab_size=26, embedding_dim=128, filter_num=128):
        super().__init__()
        self.filter_sizes = [2, 4, 8, 16, 32]
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Sequential(nn.Conv2d(1, filter_num, (fsz, embedding_dim), padding=(fsz // 2, 0)),
                          nn.BatchNorm2d(filter_num))
            for fsz in self.filter_sizes
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(len(self.filter_sizes) * filter_num, 128)
        self.classification = nn.Linear(128, 2)

    def forward(self, x):
        x = self.embedding(x).view(x.size(0), 1, x.size(1), -1)
        outs = []
        for conv in self.convs:
            o = F.relu(conv(x))
            o = F.max_pool2d(o, kernel_size=(o.size(2), o.size(3)))
            outs.append(o.view(o.size(0), -1))
        x = self.dropout(torch.cat(outs, 1))
        rep = self.linear(x)
        return self.classification(rep)


class _DS(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), int(self.y[i])


def adapter(seed: int, gpu: int = 0, fold: int = 1, epochs: int = 100, **kwargs):
    dev = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")

    class Adapter:
        def fit(self, df_train: pd.DataFrame):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            seqs = df_train["sequence"].tolist()
            self.max_len = max(len(s) for s in seqs) + 1  # [CLS] + up to max_len residues
            X = encode(seqs, self.max_len)
            y = df_train["label"].to_numpy(int)
            self.model = Net().to(dev)
            opt = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0.0025)
            sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
            loader = DataLoader(_DS(X, y), batch_size=32, shuffle=True, drop_last=False)
            self.model.train()
            for _ in range(epochs):
                for xb, yb in loader:
                    xb, yb = xb.to(dev), yb.to(dev)
                    logits = self.model(xb)
                    loss = F.cross_entropy(logits, yb)
                    loss = (loss - 0.02).abs() + 0.02  # loss flooding b=0.02
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                sched.step()
            self.model.eval()
            return self

        @torch.no_grad()
        def _proba(self, X: np.ndarray) -> np.ndarray:
            loader = DataLoader(_DS(X, np.zeros(len(X), int)), batch_size=256)
            ps = []
            for xb, _ in loader:
                ps.append(F.softmax(self.model(xb.to(dev)), dim=1)[:, 1].cpu().numpy())
            return np.concatenate(ps)

        def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
            return self._proba(encode(df["sequence"].tolist(), self.max_len))

    return Adapter()


META = {
    "citation": "LysePred (github.com/lincubator/LysePred), multi-scale CNN",
    "implementation": "ported from upstream model/LysePred.py",
    "hyperparams": "embed128, kernels [2,4,8,16,32]x128ch+BN, dropout0.2, Adam lr1e-4 wd2.5e-3, "
                   "batch32, 100 epochs, CE+flooding(0.02), CosineAnnealingWarmRestarts(10,2,1e-6)",
    "deviations": "per-fold torch seeding (upstream unseeded); final-epoch model instead of "
                  "upstream best-test-epoch checkpoint",
}
