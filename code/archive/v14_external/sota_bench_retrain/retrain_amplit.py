#!/usr/bin/env python3
"""Retrain AMPLiT (Chen et al., Nat Commun 17:1788, 2026) on the TIGER
restricted training universe.

Faithful PyTorch port of the released training code
(github.com/ChenSizhe13893461199/AMPLiT — Train_AMP_identifier2.py + utils1.py,
reference copies saved alongside this script as amplit_*_reference.py):

Three input branches, each = encoder1 (transformer block: MHA 9 heads +
FFN-128 + 2 LayerNorms) -> Conv1D(16, k=1) -> denseblock(9 layers, growth 16,
kernel 15/75/15) -> dual transition (avgpool5 / maxpool5) -> concat ->
sigmoid SE gate -> denseblock (branches 1-2):
  branch 1  one-hot  (50 x 20)
  branch 2  word2vec per-AA embeddings (50 x 100; trained on OUR corpus with
             gensim, vector_size=100 — the paper's word2vec11.bin is their
             corpus embedding, retrained here per the retraining protocol)
  branch 3  iProPy GetProDes 1547-dim descriptor folded to 50 x 31
Fusion: concat along sequence axis -> CBAM (channel MLP attention + spatial
1-D attention) -> Flatten -> Dense(16, relu) -> Dropout -> Dense(2, softmax).

Training protocol (their script): Adam lr 1e-4 (betas 0.9/0.999, eps 1e-8),
binary cross-entropy, batch 512, 21 epochs. Retrained on our universe:
positives = 4,929 restricted AMPs, negatives = the shared UniProt non-AMP pool.

Disclosed adaptations (Keras->PyTorch): (a) CBAM spatial attention is
implemented in its 1-D form (the released 2D code uses axis=3 on 3-D tensors,
a 2D-image leftover); (b) MHA head counts adjusted to divide the branch dims
(Keras allows num_heads=9 on any dim); (c) the iProPy 1547-dim descriptor is
folded to 50x31 (not 91x17) so all three branches share the sequence length,
and fusion concatenates along the channel axis — the released code's
length/channel bookkeeping (50/50/91 branches) cannot be reproduced as
written; (d) early stopping added on a 10% validation split.

Output: runs_review/benchmark_restricted/sota_amplit_preds.csv (P(AMP), 208
external test peptides).
"""
import sys
from pathlib import Path

ROOT = Path('/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER_v2/deltaMIC')
TIGER = ROOT
sys.path.insert(0, str(TIGER))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

OUT = TIGER / 'runs_review/benchmark_restricted'
CKPT_DIR = OUT / 'sota_models/amplit'
AA = 'ACDEFGHIKLMNPQRSTVWY'
AA2I = {c: i for i, c in enumerate(AA)}
MAX_LEN = 50
FP_L, FP_W = 50, 31  # 1547-dim iProPy descriptor folded to 50x31 (1547+3 zeros)

# ---- their hyperparameters (Train_AMP_identifier2.py) ----
NB_LAYERS, NB_FILTER, GROWTH = 9, 16, 16
K1, K2, K3 = 15, 75, 15
DROPOUT, WD = 0.2, 1e-6
LR, BATCH, EPOCHS = 1e-4, 512, 21
NUM_HEADS, DFF, ATT_RATE = 9, 128, 0.1


def featurize(seqs):
    """One-hot (N,50,20) + word2vec (N,50,100) + iProPy (N,91,17)."""
    from gensim.models import Word2Vec
    X1 = np.zeros((len(seqs), MAX_LEN, 20), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:MAX_LEN]):
            if c in AA2I:
                X1[i, j, AA2I[c]] = 1.0
    corpus = [[c for c in s] for s in seqs]
    w2v = Word2Vec(sentences=corpus, vector_size=100, min_count=1, seed=42, workers=8)
    X2 = np.zeros((len(seqs), MAX_LEN, 100), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:MAX_LEN]):
            if c in w2v.wv:
                X2[i, j] = w2v.wv[c]
    from propy import PyPro
    X3 = np.zeros((len(seqs), FP_L, FP_W), dtype=np.float32)
    for i, s in enumerate(seqs):
        vals = None
        # propy's PseAAC divides by (len-lambda): short sequences need reduced lambda
        for lam in (None, 4, 2, 1, 0):
            try:
                kw = {} if lam is None else {'paac_lamda': lam, 'apaac_lamda': lam,
                                             'socn_maxlag': max(lam, 1), 'qso_maxlag': max(lam, 1)}
                vals = np.asarray(list(PyPro.GetProDes(s).GetALL(**kw).values()), dtype=np.float64)
                break
            except Exception:
                continue
        if vals is None:
            continue
        n = min(len(vals), FP_L * FP_W)
        X3[i].reshape(-1)[:n] = vals[:n].astype(np.float32)
    return X1, X2, X3


class Encoder1(nn.Module):
    """utils1.encoder1: MHA(num_heads=9, key_dim=d_model) + FFN-128 + Add&Norm x2."""

    def __init__(self, d_model):
        super().__init__()
        self.qkv = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.out = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, DFF), nn.ReLU(), nn.Linear(DFF, d_model))
        self.n1 = nn.LayerNorm(d_model, eps=1e-6)
        self.n2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(ATT_RATE)

    def forward(self, x):
        d = x.size(-1)
        # Keras MHA allows num_heads=9 with key_dim=d; PyTorch needs divisibility.
        # Use the largest divisor of d that is <= 9 (5/9/1 heads for d=20/100/17).
        heads = next((h for h in range(min(NUM_HEADS, d), 0, -1) if d % h == 0), 1)
        q, k, v = [l(x).view(x.size(0), x.size(1), heads, -1).transpose(1, 2)
                   for l in self.qkv]
        att = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0)
        att = att.transpose(1, 2).reshape(x.size(0), x.size(1), d)
        h = self.n1(x + self.drop(self.out(att)))
        return self.n2(h + self.ffn(h))


class ConvFactory(nn.Module):
    def __init__(self, c_in, c_out, k):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, k, padding=k // 2, bias=False)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.drop(F.relu(self.conv(x)))


class Transition(nn.Module):
    def __init__(self, c, pool):
        super().__init__()
        self.conv = nn.Conv1d(c, NB_FILTER, 1, padding=0, bias=False)
        self.drop = nn.Dropout(DROPOUT)
        self.pool = nn.AvgPool1d(5, stride=1, padding=2) if pool == 'avg' else nn.MaxPool1d(5, stride=1, padding=2)

    def forward(self, x):
        return self.pool(self.drop(F.relu(self.conv(x))))


class DenseBlock(nn.Module):
    """DenseNet block: layer i sees c_in + i*growth channels, appends growth."""

    def __init__(self, c_in, k):
        super().__init__()
        self.layers = nn.ModuleList(
            [ConvFactory(c_in + i * GROWTH, GROWTH, k) for i in range(NB_LAYERS)])
        self.c_out = c_in + GROWTH * NB_LAYERS

    def forward(self, x):
        feats = [x]
        for lyr in self.layers:
            out = lyr(torch.cat(feats, dim=1))
            feats.append(out)
        return torch.cat(feats, dim=1)


class Branch(nn.Module):
    """One AMPLiT input branch (encoder1 -> conv -> denseblock -> dual transition -> SE gate -> denseblock)."""

    def __init__(self, d_in, k, second_block=True):
        super().__init__()
        self.enc = Encoder1(d_in)
        self.conv = nn.Conv1d(d_in, NB_FILTER, 1, bias=False)
        self.db1 = DenseBlock(NB_FILTER, k)
        c1 = NB_FILTER + GROWTH * NB_LAYERS
        self.tr_a = Transition(c1, 'avg')
        self.tr_m = Transition(c1, 'max')
        c2 = NB_FILTER  # transitions output NB_FILTER channels each
        self.gate_fc1 = nn.Linear(c2, c2 // 4)
        self.gate_fc2 = nn.Linear(c2 // 4, c2)
        self.second_block = second_block
        if second_block:
            self.db2 = DenseBlock(2 * c2, k)

    def forward(self, x):
        # x: (B, L, D)
        h = self.enc(x).transpose(1, 2)                       # (B, D, L)
        h = F.relu(self.conv(h))
        h = self.db1(h)
        xa, xb = self.tr_a(h), self.tr_m(h)
        x0 = torch.cat([xa, xb], dim=1)                       # (B, 2C, L)
        c = xa.size(1)
        # SE gate over the channel dim (per their Dense(channel*0.25)->relu->Dense(channel)->sigmoid)
        ga = torch.sigmoid(self.gate_fc2(F.relu(self.gate_fc1(xa.transpose(1, 2).mean(dim=1)))))
        gb = torch.sigmoid(self.gate_fc2(F.relu(self.gate_fc1(xb.transpose(1, 2).mean(dim=1)))))
        g = torch.sigmoid((ga + gb) / 2).unsqueeze(-1)        # (B, C, 1) broadcast to 2C
        h = x0 * torch.cat([g, g], dim=1)
        if self.second_block:
            h = self.db2(h)
        return h


class CBAM(nn.Module):
    """1-D CBAM: channel MLP attention + spatial conv attention (adapted from their 2D code)."""

    def __init__(self, c, ratio=8):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(c, c // ratio, bias=True), nn.ReLU(),
                                    nn.Linear(c // ratio, c, bias=True))
        self.spatial = nn.Conv1d(2, 1, 7, padding=3)

    def forward(self, x):                                     # (B, C, L)
        c = x.size(1)
        ca = torch.sigmoid(self.shared(x.mean(dim=2)) + self.shared(x.max(dim=2).values))  # (B, C)
        x = x * ca.unsqueeze(-1)
        sa = torch.sigmoid(self.spatial(torch.stack([x.mean(dim=1), x.max(dim=1).values], dim=1)))
        return x * sa


class AMPLiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = Branch(20, K1, second_block=True)
        self.b2 = Branch(100, K2, second_block=True)
        self.b3 = Branch(FP_W, K3, second_block=True)
        # fused channel count computed lazily on first forward
        self._fused = None
        self.head = None

    def _build_head(self, ch, length):
        self.cbam = CBAM(ch)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(ch * length, 16), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(16, 2))

    def forward(self, x1, x2, x3):
        h1, h2, h3 = self.b1(x1), self.b2(x2), self.b3(x3)
        h = torch.cat([h1, h2, h3], dim=1)                    # concat along channel axis
        if self.head is None:
            self._build_head(h.size(1), h.size(2))
            self.to(h.device)                                 # moves cbam+head together
        h = self.cbam(h)
        return self.head(h)


def main():
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)

    tr = pd.read_csv(OUT / 'train_restricted4929_tiger.csv')
    pos = tr.groupby('sequence', as_index=False).MIC_Escherichia_coli.mean().sequence.tolist()
    neg = pd.read_csv(ROOT / 'baseline/sota_bench/data_restricted/uniprot_negatives.csv').sequence.tolist()
    seqs = pos + neg
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
    print(f'positives={len(pos)} negatives={len(neg)}; featurizing...', flush=True)
    cache = Path('/tmp/amplit_feats.npz')
    if cache.exists():
        z = np.load(cache)
        X1, X2, X3 = z['X1'], z['X2'], z['X3']
    else:
        X1, X2, X3 = featurize(seqs)
        np.savez(cache, X1=X1, X2=X2, X3=X3)
    print('featurized', flush=True)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(seqs))
    n_va = int(0.1 * len(seqs))
    va_i, tr_i = idx[:n_va], idx[n_va:]

    model = AMPLiT().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=WD)

    def batchify(ids):
        return (torch.from_numpy(X1[ids]), torch.from_numpy(X2[ids]),
                torch.from_numpy(X3[ids]), torch.from_numpy(y[ids]))

    Xv = batchify(va_i)
    best_val, best_state = float('inf'), None
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = np.random.permutation(len(tr_i))
        losses = []
        for i in range(0, len(tr_i), BATCH):
            b = tr_i[perm[i:i + BATCH]]
            xb1, xb2, xb3, yb = batchify(b)
            xb1, xb2, xb3, yb = xb1.to(device), xb2.to(device), xb3.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb1, xb2, xb3)
            loss = F.cross_entropy(logits, yb.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            vl = float(F.cross_entropy(model(*[t.to(device) for t in Xv[:3]]),
                                       Xv[3].long().to(device)).cpu())
        print(f'ep{ep} train_loss={np.mean(losses):.4f} val_loss={vl:.4f}', flush=True)
        if vl < best_val - 1e-4:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict()}, CKPT_DIR / 'amplit_best.pt')

    pep = pd.read_csv(TIGER / 'runs_review/final11_peptides.csv')
    uniq = list(dict.fromkeys(pep.sequence))
    Q1, Q2, Q3 = featurize(uniq)
    model.eval()
    with torch.no_grad():
        probs = []
        for i in range(0, len(uniq), 256):
            out = model(torch.from_numpy(Q1[i:i+256]).to(device),
                        torch.from_numpy(Q2[i:i+256]).to(device),
                        torch.from_numpy(Q3[i:i+256]).to(device))
            probs.append(torch.softmax(out, dim=-1)[:, 1].cpu().numpy())
    probs = np.concatenate(probs)
    pd.Series(dict(zip(uniq, probs.tolist())), name='p_amp').to_csv(OUT / 'sota_amplit_preds.csv')
    print(f'saved sota_amplit_preds.csv; P(AMP) mean={probs.mean():.3f} '
          f'range [{probs.min():.3f},{probs.max():.3f}]', flush=True)


if __name__ == '__main__':
    main()
