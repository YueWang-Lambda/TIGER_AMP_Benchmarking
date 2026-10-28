"""FusionPeptide (mode=101) sequence-encoder variants, following
pipeline/02_toxicity_filter (github.com/haifangong/TIGER).

Framework (reference network.py): POAP one-hot (B, L, 21) -> bidirectional
recurrent encoder (hidden 256, 2 layers, dropout 0.1) -> last timestep (512-d);
global 10-D -> MLP(128, 3 layers, dropout 0.3); concat -> 640 -> 128 -> 64 -> 1
head (LeakyReLU 0.1 + Dropout 0.3). Variants:
  fp_rnn         q_encoder='rnn'  (reference-native, vanilla Elman RNN)
  fp_lstm        q_encoder='lstm' (reference-native)
  fp_transformer nn.TransformerEncoder (d_model 256, 8 heads, ff 512, 2 layers,
                   dropout 0.1, sinusoidal positions, last-timestep readout) —
                   reference has no transformer; implemented in the same frame
  fp_mamba       minimal pure-PyTorch Mamba block (selective SSM, d_model 128,
                   expand 2 -> d_inner 256, d_state 16, ZOH discretization, SiLU
                   gating, causal scan, last-timestep readout) — mamba-ssm needs
                   CUDA compilation, so reimplementation; same frame otherwise
  (fp_gru == the existing toxin_filter fusion_seq_glob arm, q_encoder='gru')

Protocol: our unified one (T=512 labels, same folds, train-fold F1 threshold,
BCE(pos_weight) + WeightedRandomSampler, AdamW lr 1e-3 wd 5e-4, early stop on
val AUC-ROC patience 12, max 80 epochs). Deviation: max_len 50 (reference
inference caps at 30 and would drop 31-50 aa peptides; our protocol keeps all).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from code.toxin_filter.features import ToxinDataset, calculate_global_properties
from code.toxin_filter.metrics import classification_metrics

MAX_LEN = 50


def _mlp(input_dim, hidden_dim, output_dim, num_layers, dropout):
    layers, d = [], input_dim
    for _ in range(num_layers - 1):
        layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        d = hidden_dim
    layers += [nn.Linear(d, output_dim)]
    return nn.Sequential(*layers)


class ReferenceHead(nn.Module):
    """FusionPeptide mode=101 shell: seq encoder + global MLP + reference fc."""

    def __init__(self, seq_encoder: nn.Module, seq_dim: int):
        super().__init__()
        self.q_encoder = seq_encoder
        self.g_encoder = _mlp(10, 128, 128, 3, 0.3)
        self.fc = nn.Sequential(
            nn.Linear(seq_dim + 128, 128), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, seq, globf):
        s = self.q_encoder(seq)  # (B, seq_dim)
        g = self.g_encoder(globf)
        return self.fc(torch.cat([s, g], dim=-1)).squeeze(-1)


class TransformerSeq(nn.Module):
    def __init__(self, d_model=256, nhead=8, ff=512, layers=2, dropout=0.1, max_len=MAX_LEN):
        super().__init__()
        self.proj = nn.Linear(21, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=False)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.pos.data.copy_(pe.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out_dim = d_model

    def forward(self, x):  # x: (B, L, 21)
        h = self.enc(self.proj(x) + self.pos[:, : x.size(1)])
        return h[:, -1, :]


class MambaSeq(nn.Module):
    """Minimal selective-SM (Mamba-style) block, pure PyTorch, causal scan."""

    def __init__(self, d_model=128, expand=2, d_state=16, max_len=MAX_LEN):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.in_proj = nn.Linear(21, self.d_inner * 2, bias=False)
        self.dt = nn.Linear(self.d_inner, 1, bias=True)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(1, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.out_dim = d_model

    def forward(self, x):  # x: (B, L, 21)
        Bsz, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        A = -torch.exp(self.A_log).expand(Bsz, self.d_inner, self.d_state)  # (B, di, N)
        h = x.new_zeros(Bsz, self.d_inner, self.d_state)
        last = None
        for t in range(L):
            xt = x_in[:, t]  # (B, di)
            delta = F.softplus(self.dt(xt))  # (B,1)
            Bt = self.B_proj(xt)  # (B,N)
            Ct = self.C_proj(xt)  # (B,N)
            Abar = torch.exp(delta.unsqueeze(-1) * A)  # (B,di,N)
            Bbar = delta.unsqueeze(-1) * Bt.unsqueeze(1)  # (B,1,N)
            h = Abar * h + Bbar * xt.unsqueeze(-1)  # (B,di,N)
            yt = (h * Ct.unsqueeze(1)).sum(-1) + self.D * xt  # (B,di)
            last = yt
        y = last * F.silu(z[:, -1])
        return self.out_proj(y)


def make_model(kind: str) -> nn.Module:
    if kind == "rnn":
        enc = nn.RNN(21, 256, num_layers=2, dropout=0.1, batch_first=True, bidirectional=True)
        seq_dim = 512

        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = enc

            def forward(self, x):
                return self.q(x)[0][:, -1, :]

        return ReferenceHead(Wrap(), seq_dim)
    if kind == "lstm":
        enc = nn.LSTM(21, 256, num_layers=2, dropout=0.1, batch_first=True, bidirectional=True)
        seq_dim = 512

        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = enc

            def forward(self, x):
                return self.q(x)[0][:, -1, :]

        return ReferenceHead(Wrap(), seq_dim)
    if kind == "transformer":
        enc = TransformerSeq()
        return ReferenceHead(enc, enc.out_dim)
    if kind == "mamba":
        enc = MambaSeq()
        return ReferenceHead(enc, enc.out_dim)
    raise ValueError(kind)


def adapter_factory(kind: str):
    def adapter(seed: int, gpu: int = 0, fold: int = 1, epochs: int = 80, **kwargs):
        dev = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")

        class Adapter:
            def fit(self, df_train: pd.DataFrame):
                torch.manual_seed(seed)
                np.random.seed(seed)
                self.g_raw = np.stack([calculate_global_properties(s, "", "") for s in df_train["sequence"]])
                mu, sd = self.g_raw.mean(0), self.g_raw.std(0) + 1e-8
                g = ((self.g_raw - mu) / sd).astype(np.float32)
                y = df_train["label"].to_numpy(int)
                ds = ToxinDataset(df_train["sequence"].tolist(), g, y, max_len=MAX_LEN, seq_style="poap")
                counts = np.bincount(y, minlength=2).astype(float)
                counts[counts == 0] = 1
                sampler = WeightedRandomSampler(torch.from_numpy(1.0 / counts[y]), num_samples=len(y), replacement=True)
                loader = DataLoader(ds, batch_size=64, sampler=sampler)
                self.model = make_model(kind).to(dev)
                pos_w = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32, device=dev)
                opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=5e-4)
                best_state, best_auc, bad = None, -1.0, 0
                for _ in range(epochs):
                    self.model.train()
                    for seq_t, globf, yb, _ in loader:
                        seq_t, globf, yb = seq_t.to(dev), globf.to(dev), yb.to(dev)
                        loss = F.binary_cross_entropy_with_logits(self.model(seq_t, globf), yb.view(-1), pos_weight=pos_w)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    p = self._proba(loader.dataset, y, dev)
                    auc = classification_metrics(y, (p >= 0.5).astype(int), p)["auc_roc"]
                    if np.isfinite(auc) and auc > best_auc:
                        best_auc, bad = auc, 0
                        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= 12:
                            break
                if best_state:
                    self.model.load_state_dict(best_state)
                # threshold on train probs
                tr_p = self._proba(loader.dataset, y, dev)
                self.thr = 0.5
                best_f1 = -1
                for t in np.linspace(0.05, 0.95, 19):
                    m = classification_metrics(y, (tr_p >= t).astype(int), tr_p)
                    if m["f1"] > best_f1:
                        best_f1, self.thr = m["f1"], float(t)
                self.mu, self.sd = mu, sd
                self.model.eval()
                return self

            @torch.no_grad()
            def _proba(self, ds, y, dev):
                loader = DataLoader(ds, batch_size=256, shuffle=False)
                ps = []
                self.model.eval()
                for seq_t, globf, _, _ in loader:
                    ps.append(torch.sigmoid(self.model(seq_t.to(dev), globf.to(dev))).cpu().numpy())
                return np.concatenate(ps)

            def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
                g_raw = np.stack([calculate_global_properties(s, "", "") for s in df["sequence"]])
                g = ((g_raw - self.mu) / self.sd).astype(np.float32)
                ds = ToxinDataset(df["sequence"].tolist(), g, np.zeros(len(df), int), max_len=MAX_LEN, seq_style="poap")
                return self._proba(ds, None, dev)

        return Adapter()

    return adapter


META_COMMON = {
    "citation": "FusionPeptide frame per haifangong/TIGER pipeline/02_toxicity_filter network.py (mode=101)",
    "hyperparams": "enc per variant + MLP(10-128-128x3,0.3) + fc 640/384/256->128->64->1; BCE(pos_weight)+WeightedSampler; "
                   "AdamW 1e-3/5e-4; AUC early stop 12; max 80 epochs; max_len 50 (ref caps 30)",
}
META = {
    "fp_rnn": {**META_COMMON, "implementation": "reference q_encoder='rnn': BiRNN(21,256,2,0.1), last timestep"},
    "fp_lstm": {**META_COMMON, "implementation": "reference q_encoder='lstm': BiLSTM(21,256,2,0.1), last timestep"},
    "fp_transformer": {**META_COMMON, "implementation": "TransformerEncoder(d256,h8,ff512,L2,drop0.1)+sinusoidal pos, last timestep (not in reference; same frame)"},
    "fp_mamba": {**META_COMMON, "implementation": "pure-torch Mamba block (d128, expand2, N16, ZOH, SiLU gate, causal scan; mamba-ssm unavailable) (not in reference; same frame)"},
}
