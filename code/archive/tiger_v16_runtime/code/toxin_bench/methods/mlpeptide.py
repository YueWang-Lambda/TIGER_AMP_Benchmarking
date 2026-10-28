"""MLpeptide hemolysis classifiers (Capecchi & Reymond, Chem Sci 2021).

Two adapters following the repo (reymond-group/MLpeptide):
  - mlpeptide_rf:  RandomForest(n_estimators=500, max_depth=10) over integer
                   character codes zero-padded to length 190 (their notebook 02b's
                   best classical model).
  - mlpeptide_rnn: their final hemolysis model (notebook 03b):
                   Embedding(vocab,100) -> GRU(100,400,1 layer) -> Linear(400,2)
                   -> LogSoftmax; SGD(lr=0.01, momentum=0.9), batch 20, NLLLoss,
                   grad clip(value=2), fixed 95 epochs (their selected checkpoint).

Note: upstream labels are 1 = NON-hemolytic; because we retrain from scratch on
our own labels (1 = toxin), no probability flip is needed here — the flip would
only apply when loading upstream's pretrained checkpoints (not used).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, Dataset

AA = "ACDEFGHIKLMNPQRSTVWY"
CHAR_ID = {c: i + 1 for i, c in enumerate(AA)}  # pad = 0 (upstream: CountVectorizer ids + 1)
PAD_LEN = 190


def char_codes(seqs: list[str]) -> np.ndarray:
    rows = []
    for s in seqs:
        ids = [CHAR_ID.get(ch, 0) for ch in s.upper()][:PAD_LEN]
        ids += [0] * (PAD_LEN - len(ids))
        rows.append(ids)
    return np.asarray(rows, dtype=np.int64)


def rf_adapter(seed: int, gpu: int = -1, fold: int = 1, **kwargs):
    model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=seed, n_jobs=-1)

    class Adapter:
        def fit(self, df_train: pd.DataFrame):
            model.fit(char_codes(df_train["sequence"].tolist()), df_train["label"].to_numpy(int))
            return self

        def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
            return model.predict_proba(char_codes(df["sequence"].tolist()))[:, 1]

    return Adapter()


class RNNNet(nn.Module):
    def __init__(self, vocab_size=21, embedding_size=100, hidden_size=400, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, 2)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return F.log_softmax(self.out(h[-1]), dim=1)


def rnn_adapter(seed: int, gpu: int = 0, fold: int = 1, epochs: int = 95, **kwargs):
    dev = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")

    class _DS(Dataset):
        def __init__(self, seqs, y):
            self.seqs, self.y = seqs, y

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, i):
            ids = [CHAR_ID.get(ch, 0) for ch in self.seqs[i].upper()]
            return torch.tensor(ids, dtype=torch.long), int(self.y[i])

    def collate(batch):
        seqs, ys = zip(*batch)
        lens = torch.tensor([len(s) for s in seqs])
        padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        return padded, lens, torch.tensor(ys)

    class Adapter:
        def fit(self, df_train: pd.DataFrame):
            torch.manual_seed(seed)
            np.random.seed(seed)
            seqs = df_train["sequence"].tolist()
            y = df_train["label"].to_numpy(int)
            self.model = RNNNet().to(dev)
            opt = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
            loader = DataLoader(_DS(seqs, y), batch_size=20, shuffle=True, drop_last=True, collate_fn=collate)
            self.model.train()
            for _ in range(epochs):
                for xb, lens, yb in loader:
                    xb, lens, yb = xb.to(dev), lens, yb.to(dev)
                    logp = self.model(xb, lens)
                    loss = F.nll_loss(logp, yb)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 2)
                    opt.step()
            self.model.eval()
            return self

        @torch.no_grad()
        def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
            loader = DataLoader(_DS(df["sequence"].tolist(), np.zeros(len(df), int)),
                                batch_size=256, collate_fn=collate)
            ps = []
            for xb, lens, _ in loader:
                logp = self.model(xb.to(dev), lens)
                ps.append(torch.exp(logp)[:, 1].cpu().numpy())
            return np.concatenate(ps)

    return Adapter()


META_RF = {
    "citation": "Capecchi & Reymond, Chem Sci 12:2454 (2021), MLpeptide notebook 02b",
    "implementation": "ported: RandomForest(500, depth10) over integer char codes padded to 190",
    "note": "upstream best classical hemolysis model of their notebook 02b",
}
META_RNN = {
    "citation": "Capecchi & Reymond, Chem Sci 12:2454 (2021), MLpeptide notebook 03b",
    "implementation": "ported final hemolysis classifier: GRU(em100,hi400,1layer), SGD lr0.01 mom0.9, "
                      "batch20, NLLLoss, clip2, fixed 95 epochs",
    "deviations": "fixed 95-epoch checkpoint instead of upstream best-test-epoch selection; "
                  "per-fold seeding",
}
