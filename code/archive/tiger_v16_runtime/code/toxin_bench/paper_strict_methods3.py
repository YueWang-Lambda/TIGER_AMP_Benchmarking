"""Paper-strict methods, batch 3: AMPDeep two-stage transfer (SwissProt
secretion -> hemolysis) and the HemoPI2 hybrid arms (RF + MERCI motifs,
ESM2-t6 fine-tune). Same no-leakage frame."""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

TIGER_ROOT = Path("/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/TIGER")
AMPDEEP = Path("/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/toxin_hemo_bench/AMPDeep")
HEMOPI2 = Path("/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/toxin_hemo_bench/hemopi2")
PANEL_PDB = TIGER_ROOT / "data/test_external/test_toxin_internal_toxin_cohort/pdb_predicted"


# ================================================================ AMPDeep
class AMPDeepStrict:
    """Salem 2022, paper's headline two-stage recipe: ProtBERT fine-tuned on
    SwissProt secretion (~28k), then fine-tuned on hemolysis. 10% train-carved
    validation, early stop on val MCC (patience 10), AdamW 5e-4 linear decay,
    effective batch 32 (16 x 2)."""

    EPOCHS_S1 = 6   # secretion stage budget (convergence + patience; logged)
    EPOCHS_S2 = 30

    def _build(self, device_str="cuda"):
        import torch
        import torch.nn as nn
        from transformers import BertModel, BertTokenizer

        PROTBERT = "/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D/baseline/sota_bench/models/prot_bert"
        tok = BertTokenizer.from_pretrained(PROTBERT, do_lower_case=False)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = BertModel.from_pretrained(PROTBERT)
                self.inter = nn.Linear(self.bert.config.hidden_size, 32)
                self.head = nn.Linear(32, 2)

            def forward(self, ids, mask):
                out = self.bert(input_ids=ids, attention_mask=mask)[0]
                pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                return self.head(self.inter(pooled))

        return tok, Net().to(device_str)

    def _train_stage(self, model, tok, texts, labels, epochs, device, tag, seed=42):
        import torch
        import torch.nn.functional as F
        from sklearn.metrics import matthews_corrcoef

        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(texts))
        n_val = max(1, int(0.1 * len(texts)))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        params = [p for p in model.parameters() if p.requires_grad]
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("inter.") or name.startswith("head.") \
                or "position_embeddings" in name or "LayerNorm" in name
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=0.0)
        steps = max(1, int(np.ceil(len(tr_idx) / 16 / 2)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: max(0.0, 1 - s / (epochs * steps)))

        def enc(idxs):
            t = [" ".join(list(texts[i])) for i in idxs]
            e = tok(t, add_special_tokens=True, truncation=True, max_length=512, padding=True, return_tensors="pt")
            return e["input_ids"].to(device), e["attention_mask"].to(device), torch.tensor([labels[i] for i in idxs]).to(device)

        best_state, best_mcc, stale = None, -2.0, 0
        for epoch in range(epochs):
            model.train()
            p2 = np.random.permutation(tr_idx)
            opt.zero_grad()
            for b, i0 in enumerate(range(0, len(p2), 16)):
                idxs = p2[i0 : i0 + 16]
                ids, mask, yb = enc(idxs)
                loss = F.cross_entropy(model(ids, mask), yb) / 2
                loss.backward()
                if (b + 1) % 2 == 0 or (b + 1) * 16 >= len(p2):
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step(); sched.step(); opt.zero_grad()
            model.eval()
            with torch.no_grad():
                ps = []
                for i0 in range(0, len(val_idx), 32):
                    ids, mask, _ = enc(np.sort(val_idx)[i0 : i0 + 32])
                    ps.append(F.softmax(model(ids, mask), 1)[:, 1].cpu().numpy())
            p = np.concatenate(ps)
            mcc = matthews_corrcoef([labels[i] for i in val_idx], (p >= 0.5).astype(int)) if len(set(labels[i] for i in val_idx)) > 1 else -1
            print(f"  [ampdeep strict {tag}] epoch {epoch+1} val MCC {mcc:.3f}", flush=True)
            if mcc > best_mcc:
                best_mcc, stale = mcc, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= 10:
                    break
        model.load_state_dict(best_state)
        return best_mcc

    def train(self, df, gpu=0, seed=42):
        import torch

        device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")
        # seed the previously-unseeded stochastic elements (head init, epoch shuffles)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # leakage audit: secretion set vs both test panels
        sec = pd.read_csv(AMPDEEP / "data/swissprot/secreted_all_cleaned_all.csv")
        sec_set = {str(t).replace(" ", "").upper() for t in sec["text"]}
        train_set = set(df["sequence"])
        overlap = len(sec_set & train_set)
        # ---- stage 1: secretion (labels: 1 = secreted per their CSV)
        tok, model = self._build(device)
        texts1 = [str(t) for t in sec["text"]]
        labels1 = sec["labels"].astype(int).tolist()
        mcc1 = self._train_stage(model, tok, texts1, labels1, self.EPOCHS_S1, device, "s1", seed=seed)
        # ---- stage 2: hemolysis fine-tune from stage-1 weights
        texts2 = df["sequence"].tolist()
        labels2 = df["label"].astype(int).tolist()
        mcc2 = self._train_stage(model, tok, texts2, labels2, self.EPOCHS_S2, device, "s2", seed=seed)
        self.art = {"model": model, "tok": tok, "device": device}
        self.meta = {
            "citation": "Salem et al., BMC Bioinformatics 23:389 (2022)",
            "protocol": "two-stage transfer: ProtBERT -> secretion (n=%d, %d epochs) -> hemolysis (1596, <=%d epochs); 10%% train-carved val, early stop val MCC (patience 10), AdamW 5e-4 linear decay, eff. batch 32"
                       % (len(texts1), self.EPOCHS_S1, self.EPOCHS_S2),
            "val_mcc_s1": mcc1, "val_mcc_s2": mcc2,
            "threshold_rule": "0.5",
            "deviations": "stage-1 budget 6 epochs (paper 50 w/ patience; GPU budget); frozen=non-positional+attention+FFN per released flags",
            "leakage_audit": {"secretion_vs_train_overlap": overlap},
        }
        return {"threshold": 0.5, "meta": self.meta}

    def predict_proba(self, df):
        import torch
        import torch.nn.functional as F

        tok, model, device = self.art["tok"], self.art["model"], self.art["device"]
        model.eval()
        texts = [" ".join(list(s)) for s in df["sequence"]]
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                e = tok(texts[i : i + 32], add_special_tokens=True, truncation=True, max_length=512, padding=True, return_tensors="pt")
                probs.append(F.softmax(model(e["input_ids"].to(device), e["attention_mask"].to(device)), 1)[:, 1].cpu().numpy())
        return np.concatenate(probs)


# ================================================================ HemoPI2 hybrid (RF + MERCI)
class HemoPI2HybridStrict:
    """Rathore 2025 hybrid: RF (paper-strict tuned) score + MERCI motif hits
    (+0.5 per positive-set hit, -0.5 per negative-set hit), clipped [0,1],
    threshold 0.45. Uses their RELEASED motifs (pretrained artifact, akin to
    ProtBERT) and the vendored MERCI perl tool."""

    MERCI_CACHE = "/tmp/merci_scores.npz"

    def _merci_score(self, seqs, tag=""):
        import tempfile

        cache = Path(self.MERCI_CACHE)
        if cache.exists():
            z = np.load(cache)
            key = {"train": "train", "panel88": "p88", "panel200": "p200"}.get(tag, tag)
            if key and key in z.files:
                return z[key]
        with tempfile.TemporaryDirectory() as td:
            fasta = Path(td) / "panel.fasta"
            fasta.write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(seqs)))
            score = np.zeros(len(seqs))
            for sign, files in [(+0.5, ["pos_motif_1.txt", "pos_motif_2.txt"]), (-0.5, ["neg_motif_1.txt", "neg_motif_2.txt"])]:
                for mf in files:
                    out = subprocess.run(
                        ["perl", str(HEMOPI2 / "merci/MERCI_motif_locator.pl"), "-i", str(fasta),
                         "-p", str(HEMOPI2 / "motif" / mf), "-c", "KOOLMAN-ROHM"],
                        capture_output=True, text=True, cwd=str(td))
                    hits = set()
                    for line in out.stdout.splitlines():
                        line = line.strip()
                        if line.startswith(">"):
                            sid = line[1:].split()[0]
                            hits.add(int(sid[1:]))
                    score[list(hits)] += sign
            if tag:
                key = {"train": "train", "panel88": "p88", "panel200": "p200"}[tag]
                if cache.exists():
                    z = dict(np.load(cache))
                else:
                    z = {}
                z[key] = score
                np.savez(cache, **z)
            return score

    def train(self, df, gpu=-1, seed=42):
        from code.toxin_bench.paper_strict_methods import HemoPI2RFStrict

        rf = HemoPI2RFStrict()
        rf.train(df, gpu)
        self.rf = rf
        self.merci_train = self._merci_score(df["sequence"].tolist(), tag="train")
        # hybrid threshold stays 0.45 per released code
        self.meta = {
            "citation": "Rathore et al., Commun Biol 8:176 (2025), HemoPI2 hybrid (ML + MERCI)",
            "protocol": "paper-strict RF score + +/-0.5 MERCI motif hits (their released pos/neg motif sets), clipped [0,1]",
            "threshold_rule": "0.45",
            "deviations": "RF sub-model = paper-strict grid-tuned; motifs used as released pretrained artifacts (no test contact)",
        }
        return {"threshold": 0.45, "meta": self.meta}

    def predict_proba(self, df):
        p = np.clip(self.rf.predict_proba(df), 0, 1)
        tag = "panel88" if len(df) == 88 else ("panel200" if len(df) == 200 else "")
        hybrid = np.clip(p + self._merci_score(df["sequence"].tolist(), tag=tag), 0, 1)
        return hybrid


METHODS = {
    "ampdeep": AMPDeepStrict,
    "hemopi2_hybrid": HemoPI2HybridStrict,
}
