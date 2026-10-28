"""HemoNet (Yaseen et al., JBCB 2021) — SeqVec + terminal fingerprints, per-paper.

Upstream's Results.py (model definition) is NOT published; this adapter follows
the paper's Methods exactly:
  features = SeqVec 1024-d (global average pooling over the 3 layers and the
             sequence length) ++ ECFP4 (Morgan r=2, 1024-bit) of the N-terminal
             modification SMILES ++ same for C-terminal (Free -> zeros)
  scaler: z-score over the concatenated 3072-d feature (fit on train fold)
  model:  3072 -> 2048 (ReLU) -> 1024 (tanh) -> 100 (tanh) -> 1 linear output
  training: MSE loss on {+1,-1} targets, Adam lr=1e-3, weight_decay=1e-4,
            StepLR(gamma=0.95) every 50 epochs (paper's recipe)
  probability: sigmoid of the raw score (monotone; used for ranking and the
            harness's threshold calibration; upstream used Platt scaling)

SeqVec embeddings are precomputed once by runs_toxin_bench/scripts/compute_seqvec.py
into baseline/toxin_hemo_bench/seqvec/seqvec_embeddings.npz (keyed by sequence).
Terminal modification codes (DBAASP shorthand) map to cap SMILES; unmapped/NA
-> all-zero fingerprint ("Free" upstream).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEQVEC_NPZ = Path(
    "/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/toxin_hemo_bench/seqvec/seqvec_embeddings.npz"
)

# DBAASP terminal shorthand -> cap SMILES (subset actually present in our data)
TERM_SMILES = {
    "AMD": "N", "AM": "N", "AMIDE": "N", "NH2": "N",
    "ACT": "CC(=O)N", "ACETYL": "CC(=O)N", "ACE": "CC(=O)N", "AC": "CC(=O)N",
    "ETA": "OCCN",
    "C10": "CCCCCCCCCC(=O)N", "C12": "CCCCCCCCCCCC(=O)N", "C14": "CCCCCCCCCCCCCC(=O)N",
    "C16": "CCCCCCCCCCCCCCCC(=O)N", "C16H33-NH2": "CCCCCCCCCCCCCCCC(=O)N",
    "C8": "CCCCCCCC(=O)N",
}


def fingerprint(smiles: str) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(1024)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    arr = np.zeros(1024)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def term_features(codes: pd.Series) -> np.ndarray:
    rows = []
    for v in codes:
        s = TERM_SMILES.get(str(v).strip().upper() if pd.notna(v) else "", "")
        rows.append(fingerprint(s) if s else np.zeros(1024))
    return np.asarray(rows, dtype=np.float32)


class Net(nn.Module):
    def __init__(self, in_dim=3072):
        super().__init__()
        self.h1 = nn.Linear(in_dim, 2048)
        self.h2 = nn.Linear(2048, 1024)
        self.h3 = nn.Linear(1024, 100)
        self.out = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.h1(x))
        x = torch.tanh(self.h2(x))
        x = torch.tanh(self.h3(x))
        return self.out(x).squeeze(-1)


def _seqvec(seqs: list[str]) -> np.ndarray:
    if not SEQVEC_NPZ.exists():
        raise SystemExit(
            f"Missing {SEQVEC_NPZ} — run runs_toxin_bench/scripts/compute_seqvec.py in the seqvec env first"
        )
    z = np.load(SEQVEC_NPZ)
    return np.stack([z[s] for s in seqs]).astype(np.float32)


def features(df: pd.DataFrame) -> np.ndarray:
    seqvec = _seqvec(df["sequence"].astype(str).str.upper().str.strip().tolist())
    n = term_features(df.get("n_terminus", pd.Series([""] * len(df))))
    c = term_features(df.get("c_terminus", pd.Series([""] * len(df))))
    return np.concatenate([seqvec, n, c], axis=1)


def adapter(seed: int, gpu: int = 0, fold: int = 1, epochs: int = 150, batch: int = 64, **kwargs):
    dev = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")

    class Adapter:
        def fit(self, df_train: pd.DataFrame):
            torch.manual_seed(seed)
            np.random.seed(seed)
            X = features(df_train)
            self.mu = X.mean(axis=0)
            self.sd = X.std(axis=0) + 1e-8
            Xn = torch.from_numpy(((X - self.mu) / self.sd).astype(np.float32))
            y = torch.from_numpy(np.where(df_train["label"].to_numpy(int) == 1, 1.0, -1.0).astype(np.float32))
            self.model = Net(X.shape[1]).to(dev)
            opt = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.95)
            n = len(Xn)
            for _ in range(epochs):
                perm = torch.randperm(n)
                for i in range(0, n, batch):
                    idx = perm[i : i + batch]
                    xb, yb = Xn[idx].to(dev), y[idx].to(dev)
                    loss = nn.functional.mse_loss(self.model(xb), yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                sched.step()
            self.model.eval()
            return self

        @torch.no_grad()
        def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
            X = features(df)
            Xn = torch.from_numpy(((X - self.mu) / self.sd).astype(np.float32))
            scores = []
            self.model.eval()
            for i in range(0, len(Xn), 256):
                scores.append(self.model(Xn[i : i + 256].to(dev)).cpu().numpy())
            s = np.concatenate(scores)
            return 1.0 / (1.0 + np.exp(-s))  # sigmoid of the raw score

    return Adapter()


META = {
    "citation": "Yaseen et al., J Bioinform Comput Biol 19:2150021 (2021), HemoNet "
                "(github.com/adibayaseen/HemoNet)",
    "implementation": "per-paper reimplementation (upstream Results.py unpublished): SeqVec1024(GAP) "
                      "+ ECFP4(1024)x2 terminal fingerprints -> 2048(ReLU)/1024(tanh)/100(tanh)/1",
    "hyperparams": "MSE on +/-1 targets, Adam lr1e-3 wd1e-4, StepLR(50, 0.95), z-scored features",
    "deviations": "batch=64 & 150 fixed epochs (paper unspecified; early stopping on train loss); "
                  "probability = sigmoid(raw score) instead of Platt scaling; terminal codes mapped "
                  "to cap SMILES (unmapped -> zero fingerprint)",
    "weights": "SeqVec LM: baseline/toxin_hemo_bench/seqvec/weights.hdf5 (pretrained feature extractor "
               "only); classifier trained from scratch",
}
