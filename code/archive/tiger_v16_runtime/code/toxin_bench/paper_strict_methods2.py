"""Paper-strict methods, batch 2: MLpeptide (full grids + FPR<=5% threshold)
and Plisson (screening -> GridSearchCV -> feature-reduction variants).
Same no-leakage frame as paper_strict_methods.py."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def char_codes(seqs):
    from code.toxin_bench.methods.mlpeptide import char_codes as _f
    return _f(seqs)


def fpr05_threshold(y, score):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, score)
    ok = np.where(fpr <= 0.05)[0]
    if len(ok) == 0:
        return float(thr[0])
    return float(thr[ok[np.argmax(tpr[ok])]])


# ================================================================ MLpeptide-RF
class MLpeptideRFStrict:
    """Capecchi 2021 notebook 02b: GridSearchCV(RandomForest, their grid,
    scoring=roc_auc) then final fit; threshold = ROC point with FPR<=5% on train."""

    GRID = {"max_depth": [10, 30, 50, 70, 90, None],
            "n_estimators": [10, 100, 250, 500, 750, 1000, 1500, 2000]}

    def train(self, df, gpu=-1, seed=42):
        X = char_codes(df["sequence"].tolist())
        y = df["label"].to_numpy(int)
        grid = GridSearchCV(
            RandomForestClassifier(random_state=seed, n_jobs=-1), self.GRID,
            cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc", n_jobs=-1,
        )
        grid.fit(X, y)
        self.model = grid.best_estimator_
        train_prob = self.model.predict_proba(X)[:, 1]
        thr = fpr05_threshold(y, train_prob)
        self.meta = {
            "citation": "Capecchi et al., Chem Sci 12:9221 (2021), notebook 02b",
            "protocol": "GridSearchCV(RF, their grid, roc_auc, 5-fold) -> refit; threshold = FPR<=5% ROC point on train",
            "best_params": grid.best_params_, "cv_auc": grid.best_score_,
            "threshold_rule": f"FPR<=5% on train ROC -> {thr:.4f}",
            "deviations": "sklearn 1.4 vs 0.22 defaults; threshold on train (their notebook used the test split - leakage guard)",
        }
        return {"threshold": thr, "meta": self.meta}

    def predict_proba(self, df):
        return self.model.predict_proba(char_codes(df["sequence"].tolist()))[:, 1]


# ================================================================ MLpeptide-RNN
class MLpeptideRNNStrict:
    """Capecchi 2021 notebook 03b: 60-config grid (embedding x hidden x layers),
    each trained on an internal 75/25 re-split, selected by eval ROC; the best
    config is retrained on the full training set; FPR<=5% threshold."""

    EMB = [2, 21, 42, 100]
    HID = [50, 100, 200, 300, 400]
    LAY = [1, 2, 3]
    EPOCHS = 50  # their protocol trains to 150 with post-hoc best-epoch; reduced (logged)

    def train(self, df, gpu=0, seed=42):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset

        dev = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")
        seqs = df["sequence"].tolist()
        y = df["label"].to_numpy(int)

        class _DS(Dataset):
            def __init__(self, seqs, ys):
                self.seqs, self.ys = seqs, ys

            def __len__(self):
                return len(self.seqs)

            def __getitem__(self, i):
                ids = [ord(c) % 21 + 1 for c in self.seqs[i].upper()]
                return torch.tensor(ids, dtype=torch.long), int(self.ys[i])

        def collate(batch):
            ss, ys = zip(*batch)
            lens = torch.tensor([len(s) for s in ss])
            pad = nn.utils.rnn.pad_sequence(ss, batch_first=True)
            return pad, lens, torch.tensor(ys)

        class RNN(nn.Module):
            def __init__(self, emb, hid, lay):
                super().__init__()
                self.emb = nn.Embedding(22, emb, padding_idx=0)
                self.gru = nn.GRU(emb, hid, num_layers=lay, batch_first=True)
                self.out = nn.Linear(hid, 2)

            def forward(self, x, lens):
                packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lens, batch_first=True, enforce_sorted=False)
                _, h = self.gru(packed)
                return F.log_softmax(self.out(h[-1]), dim=1)

        # ---- stage 1: 60-config search on internal 75/25 split (train-only)
        rng = np.random.RandomState((seed - 42) % (2**32))
        perm = rng.permutation(len(seqs))
        n_quarter = len(seqs) // 4
        q_idx, f_idx = perm[:n_quarter], perm[n_quarter:]
        best = None
        configs = [(e, h, l) for e in self.EMB for h in self.HID for l in self.LAY]
        for ci, (e, h, l) in enumerate(configs):
            torch.manual_seed(seed)
            model = RNN(e, h, l).to(dev)
            opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            loader = DataLoader(_DS([seqs[i] for i in f_idx], y[f_idx]), batch_size=20, shuffle=True,
                                drop_last=True, collate_fn=collate)
            for _ in range(self.EPOCHS):
                model.train()
                for xb, lens, yb in loader:
                    xb, lens, yb = xb.to(dev), lens, yb.to(dev)
                    loss = F.nll_loss(model(xb, lens), yb)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), 2)
                    opt.step()
            model.eval()
            from sklearn.metrics import roc_auc_score
            with torch.no_grad():
                loader = DataLoader(_DS([seqs[i] for i in q_idx], y[q_idx]), batch_size=256, collate_fn=collate)
                p = np.concatenate([torch.exp(model(xb.to(dev), lens)[:, 1]).cpu().numpy() for xb, lens, _ in loader])
            try:
                auc = roc_auc_score(y[q_idx], p)
            except ValueError:
                auc = 0.5
            if best is None or auc > best[0]:
                best = (auc, (e, h, l))
            if (ci + 1) % 10 == 0:
                print(f"  [mlpeptide_rnn strict] config {ci+1}/60 best AUC {best[0]:.3f} @ {best[1]}", flush=True)
        auc, (e, h, l) = best

        # ---- stage 2: retrain best config on the FULL training table
        torch.manual_seed(seed)
        model = RNN(e, h, l).to(dev)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loader = DataLoader(_DS(seqs, y), batch_size=20, shuffle=True, drop_last=True, collate_fn=collate)
        for _ in range(self.EPOCHS):
            model.train()
            for xb, lens, yb in loader:
                xb, lens, yb = xb.to(dev), lens, yb.to(dev)
                loss = F.nll_loss(model(xb, lens), yb)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 2)
                opt.step()
        model.eval()
        self.model, self.dev, self.cfg = model, dev, (e, h, l)
        with torch.no_grad():
            loader = DataLoader(_DS(seqs, y), batch_size=256, collate_fn=collate)
            tr_p = np.concatenate([torch.exp(model(xb.to(dev), lens)[:, 1]).cpu().numpy() for xb, lens, _ in loader])
        thr = fpr05_threshold(y, tr_p)
        self._collate = collate
        self._DS = _DS
        self.meta = {
            "citation": "Capecchi et al., Chem Sci 12:9221 (2021), notebook 03b",
            "protocol": "60-config grid (emb x hid x layers) on internal 75/25 split (eval ROC), best refit on full train; SGD 0.01/0.9, batch 20, clip 2",
            "best_config": {"emb": e, "hid": h, "layers": l, "search_auc": auc},
            "threshold_rule": f"FPR<=5% on train ROC -> {thr:.4f}",
            "deviations": "50 epochs (their 150 + best-epoch post-hoc); search split seeded; threshold on train (their notebook used test)",
        }
        return {"threshold": thr, "meta": self.meta}

    def predict_proba(self, df):
        import torch
        from torch.utils.data import DataLoader
        self.model.eval()
        with torch.no_grad():
            loader = DataLoader(self._DS(df["sequence"].tolist(), np.zeros(len(df), int)), batch_size=256,
                                collate_fn=self._collate)
            return np.concatenate([torch.exp(self.model(xb.to(self.dev), lens)[:, 1]).cpu().numpy() for xb, lens, _ in loader])


# ================================================================ Plisson
class PlissonStrict:
    """Plisson 2020: (1) 10-fold CV screening of 14 classifiers on 56 modlamp
    descriptors, (2) GridSearchCV on the top models with their grids,
    (3) feature-reduction variants; selection = accuracy. Final = best variant."""

    def _features(self, df):
        from code.toxin_bench.methods.plisson import modlamp_features
        X = modlamp_features(df["sequence"].tolist())
        mu, sd = X.min(0), X.max(0)
        rng = np.where(sd - mu < 1e-12, 1.0, sd - mu)
        self._mu, self._rng = mu, rng
        return (X - mu) / rng

    def _norm(self, df):
        from code.toxin_bench.methods.plisson import modlamp_features
        X = modlamp_features(df["sequence"].tolist())
        return (X - self._mu) / self._rng

    def train(self, df, gpu=-1, seed=42):
        X = self._features(df)
        y = df["label"].to_numpy(int)
        cv10 = StratifiedKFold(10, shuffle=True, random_state=seed)
        screening = {
            "LR": Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=2000, random_state=seed))]),
            "KNN": KNeighborsClassifier(),
            "CART": DecisionTreeClassifier(random_state=seed),
            "RF": RandomForestClassifier(random_state=seed, max_depth=5, n_jobs=-1),
            "GBC": GradientBoostingClassifier(random_state=seed),
            "AdaBoost": AdaBoostClassifier(random_state=seed),
            "LDA": LinearDiscriminantAnalysis(),
            "QDA": QuadraticDiscriminantAnalysis(),
            "GaussianNB": GaussianNB(),
            "SVC-lin": Pipeline([("s", StandardScaler()), ("m", SVC(kernel="linear", probability=True, random_state=seed))]),
            "SVC-rbf": Pipeline([("s", StandardScaler()), ("m", SVC(kernel="rbf", probability=True, random_state=seed))]),
            "SVC-poly": Pipeline([("s", StandardScaler()), ("m", SVC(kernel="poly", probability=True, random_state=seed))]),
            "SVC-sig": Pipeline([("s", StandardScaler()), ("m", SVC(kernel="sigmoid", probability=True, random_state=seed))]),
        }
        scores = {}
        for name, m in screening.items():
            accs = []
            for tr, va in cv10.split(X, y):
                mm = copy.deepcopy(m)
                mm.fit(X[tr], y[tr])
                accs.append(accuracy_score(y[va], mm.predict(X[va])))
            scores[name] = float(np.mean(accs))
        top = sorted(scores, key=scores.get, reverse=True)[:3]
        grids = {
            "GBC": {"n_estimators": [100, 200, 240], "max_depth": [3, 4], "min_samples_leaf": [2, 10], "max_features": ["sqrt"]},
            "RF": {"n_estimators": [100, 208], "max_depth": [10, 14, None], "min_samples_leaf": [2, 4], "max_features": ["sqrt", "log2"]},
            "SVC-rbf": {"m__C": [1, 10, 100], "m__gamma": ["scale", 0.01, 0.1]},
            "LR": {"m__C": [1, 100, 1000]},
        }
        best = None
        for name in top:
            base = screening[name]
            g = grids.get(name)
            if g is None:
                acc = scores[name]
                params = {}
            else:
                gs = GridSearchCV(base, g, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="accuracy", n_jobs=-1)
                gs.fit(X, y)
                base, acc, params = gs.best_estimator_, gs.best_score_, gs.best_params_
            if best is None or acc > best[0]:
                best = (acc, name, base, params)
        acc, name, model, params = best
        self.model = model
        self.meta = {
            "citation": "Plisson et al., Sci Rep 10:16581 (2020)",
            "protocol": "10-fold screening (13 classifiers, accuracy) -> GridSearchCV on top-3 -> best refit; 56 modlamp descriptors, min-max on train",
            "screening_scores": {k: round(v, 4) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:6]},
            "selected_model": name, "best_params": params, "cv_acc": acc,
            "threshold_rule": "0.5",
            "deviations": "13 of their ~14 classifiers (XGBoost comparison skipped); feature-reduction variants (RFECV/r>0.75 trim) skipped - selection already train-internal; min-max stats from full train (upstream used per-split stats)",
        }
        return {"threshold": 0.5, "meta": self.meta}

    def predict_proba(self, df):
        return self.model.predict_proba(self._norm(df))[:, 1]


METHODS = {
    "mlpeptide_rf": MLpeptideRFStrict,
    "mlpeptide_rnn": MLpeptideRNNStrict,
    "plisson": PlissonStrict,
}
