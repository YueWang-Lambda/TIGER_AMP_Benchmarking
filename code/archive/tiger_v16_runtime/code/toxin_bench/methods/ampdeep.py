"""AMPDeep (Salem et al., BMC Bioinformatics 2022) — ProtBERT fine-tune, ported.

Follows the released training.py (standalone variant, transfer=0):
  local ProtBERT (bert_bfd) + head Linear(1024,32) -> Linear(32,2), no activation,
  mean pooling over the attention mask (includes CLS/SEP); everything frozen
  except position embeddings, all LayerNorms, and the head; unweighted CE;
  AdamW lr 5e-4 with linear decay (no warmup), grad clip 1.0; effective batch 32
  (16 x 2 accumulation); max 50 epochs with early stopping (patience 10) on a
  10%-of-train validation split's MCC; spaced-residue tokenization.

Paper's headline models additionally fine-tune from a secretion-prediction
stage; the released script defaults to the standalone variant reproduced here
(noted in META).
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROTBERT_PATH = "/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/sota_bench/models/prot_bert"


class Net(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.0)
        self.inter = nn.Linear(bert.config.hidden_size, 32)
        self.classifier = nn.Linear(32, 2)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooled = (out * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        return self.classifier(self.inter(self.dropout(pooled)))


class _DS(Dataset):
    def __init__(self, seqs, y):
        self.seqs, self.y = seqs, y

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], int(self.y[i])


def adapter(seed: int, gpu: int = 0, fold: int = 1, epochs: int = 50, **kwargs):
    from transformers import BertModel, BertTokenizer

    dev = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = BertTokenizer.from_pretrained(PROTBERT_PATH, do_lower_case=False)
    bert = BertModel.from_pretrained(PROTBERT_PATH)
    model = Net(bert).to(dev)

    # Freeze everything except position embeddings, LayerNorms, and the head.
    for name, p in model.named_parameters():
        trainable = (
            name.startswith("inter.")
            or name.startswith("classifier.")
            or "position_embeddings" in name
            or "LayerNorm" in name
        )
        p.requires_grad = trainable

    def tokenize(seqs):
        return [" ".join(list(s.upper())) for s in seqs]

    class Adapter:
        def fit(self, df_train: pd.DataFrame):
            seqs = tokenize(df_train["sequence"].tolist())
            y = df_train["label"].to_numpy(int)
            rng = np.random.RandomState(42)
            perm = rng.permutation(len(seqs))
            n_val = max(1, int(0.1 * len(seqs)))
            val_idx, tr_idx = perm[:n_val], perm[n_val:]

            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.0)
            steps_per_epoch = max(1, int(np.ceil(len(tr_idx) / 16 / 2)))
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: max(0.0, 1 - step / (epochs * steps_per_epoch))
            )

            def batches(idxs):
                for i in range(0, len(idxs), 16):
                    yield idxs[i : i + 16]

            def encode_batch(idxs):
                texts = [seqs[i] for i in idxs]
                enc = tokenizer(texts, add_special_tokens=True, truncation=True, max_length=512, padding=True, return_tensors="pt")
                return enc["input_ids"].to(dev), enc["attention_mask"].to(dev), torch.tensor([y[i] for i in idxs]).to(dev)

            from sklearn.metrics import matthews_corrcoef

            best_state, best_mcc, stale = None, -2.0, 0
            for epoch in range(epochs):
                model.train()
                tr_perm = np.random.permutation(tr_idx)
                opt.zero_grad()
                for b, idxs in enumerate(batches(tr_perm)):
                    ids, mask, yb = encode_batch(idxs)
                    loss = F.cross_entropy(model(ids, mask), yb) / 2  # accumulation 2
                    loss.backward()
                    if (b + 1) % 2 == 0 or (b + 1) * 16 >= len(tr_perm):
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        opt.step()
                        sched.step()
                        opt.zero_grad()
                model.eval()
                with torch.no_grad():
                    probs = []
                    for idxs in batches(np.sort(val_idx)):
                        ids, mask, _ = encode_batch(idxs)
                        probs.append(F.softmax(model(ids, mask), dim=1)[:, 1].cpu().numpy())
                probs = np.concatenate(probs)
                mcc = matthews_corrcoef(y[val_idx], (probs >= 0.5).astype(int)) if len(np.unique(y[val_idx])) > 1 else -1.0
                if mcc > best_mcc:
                    best_mcc, stale = mcc, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    stale += 1
                    if stale >= 10:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            return self

        @torch.no_grad()
        def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
            texts = tokenize(df["sequence"].tolist())
            probs = []
            model.eval()
            for i in range(0, len(texts), 16):
                enc = tokenizer(texts[i : i + 16], add_special_tokens=True, truncation=True, max_length=512,
                                padding=True, return_tensors="pt")
                logits = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
                probs.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
            return np.concatenate(probs)

    return Adapter()


META = {
    "citation": "Salem et al., BMC Bioinformatics 23:389 (2022), AMPDeep (github.com/milad73s/AMPDeep)",
    "implementation": "standalone ProtBERT fine-tune (released default, transfer=0): frozen except "
                      "pos-emb + LayerNorm + head; head 1024-32-2, mean pooling",
    "hyperparams": "AdamW lr 5e-4 linear decay no warmup, eff batch 32 (16x2), clip 1.0, max 50 ep, "
                   "early stop patience 10 on 10%-train val MCC, unweighted CE",
    "deviations": "paper headline adds secretion-task transfer stage; released default (reproduced "
                  "here) initializes from plain ProtBERT-BFD",
    "weights": "local ProtBERT: baseline/sota_bench/models/prot_bert",
}
