"""Paper-strict method implementations (batch 1: hemopi1, hemopred, hemonet,
lysepred, hemopi2_rf). Each returns a trained object with
.train(df_train, gpu) -> {"threshold": float, "meta": {...}} and
.predict_proba(df) -> np.ndarray.

All selection is train-internal; deviations forced by the no-leakage rule are
recorded in meta["deviations"].
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

AA = "ACDEFGHIKLMNPQRSTVWY"
AIDX = {a: i for i, a in enumerate(AA)}


# ---------------------------------------------------------------- features
def aac_dpc(seqs):
    from code.toxin_bench.methods.hemopi1 import aac_dpc as _f
    return _f(seqs)


def binary_profiles(seqs, term_len=15):
    rows = []
    for s in seqs:
        v = np.zeros(2 * term_len * 20)
        n = s[:term_len]
        c = s[-term_len:][::-1]
        for i, ch in enumerate(n):
            if ch in AIDX:
                v[i * 20 + AIDX[ch]] = 1
        for i, ch in enumerate(c):
            if ch in AIDX:
                v[(term_len + i) * 20 + AIDX[ch]] = 1
        rows.append(v)
    return np.asarray(rows)


# ---------------------------------------------------------------- helpers
def cv_acc(model, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, va in skf.split(X, y):
        m = copy.deepcopy(model)
        m.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[va], m.predict(X[va])))
    return float(np.mean(accs))


def fpr05_threshold(y, score):
    """MLpeptide rule: ROC operating point with FPR <= 5% (max TPR under it)."""
    fpr, tpr, thr = roc_curve(y, score)
    ok = fpr <= 0.05
    if not ok.any():
        return float(thr[0])
    i = np.argmax(tpr[ok])
    return float(np.asarray(thr)[ok][i if False else np.where(ok)[0][i]])


# ================================================================ HemoPI-1
class HemoPI1Strict:
    """Chaudhary 2016: 5-fold CV screening of SVM(light)/IBK/MLP/Logistic/J48/RF
    over AAC/DPC/binary-profile features; SVM hyperparameters via CV grid."""

    def train(self, df, gpu=-1, seed=42):
        seqs = df["sequence"].tolist()
        y = df["label"].to_numpy(int)
        feats = {"AAC+DPC": aac_dpc(seqs), "AAC+DPC+BinProf": np.concatenate([aac_dpc(seqs), binary_profiles(seqs)], axis=1)}
        models = {
            "SVM-rbf": Pipeline([("s", StandardScaler()), ("m", SVC(kernel="rbf", probability=True, random_state=seed))]),
            "KNN(IBK)": KNeighborsClassifier(n_neighbors=1),
            "MLP": Pipeline([("s", StandardScaler()), ("m", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed))]),
            "Logistic": Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=2000, random_state=seed))]),
            "J48(CART)": DecisionTreeClassifier(random_state=seed),
            "RF": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        }
        best = None
        for fname, X in feats.items():
            for mname, m in models.items():
                acc = cv_acc(m, X, y, seed=seed)
                if best is None or acc > best[0]:
                    best = (acc, fname, mname)
        acc, fname, mname = best
        X = feats[fname]
        if mname == "SVM-rbf":
            grid = GridSearchCV(
                Pipeline([("s", StandardScaler()), ("m", SVC(kernel="rbf", probability=True, random_state=seed))]),
                {"m__C": [1, 10, 100], "m__gamma": [1e-3, 1e-2, 1e-1]},
                cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="accuracy", n_jobs=-1,
            )
            grid.fit(X, y)
            self.model = grid.best_estimator_
            self.meta = {"selected_feature": fname, "selected_model": mname, "cv_acc_screen": acc,
                         "svm_best": {k: v for k, v in grid.best_params_.items()}, "svm_cv_acc": grid.best_score_}
        else:
            self.model = copy.deepcopy(models[mname])
            self.model.fit(X, y)
            self.meta = {"selected_feature": fname, "selected_model": mname, "cv_acc_screen": acc}
        self.meta.update({
            "citation": "Chaudhary et al., Sci Rep 6:22843 (2016)",
            "protocol": "5-fold CV screening (SVM/IBK/MLP/Logistic/J48/RF) x features (AAC/DPC/binary profile), SVM C/gamma grid; selection = mean CV accuracy",
            "threshold_rule": "0.5 (SVM Platt probability)",
            "deviations": "SVM_light->sklearn SVC; IBK k=1; binary profile N15+C15; screening criterion = accuracy per paper-era convention",
        })
        return {"threshold": 0.5, "meta": self.meta}

    def predict_proba(self, df):
        X = None  # recompute using stored feature mode
        seqs = df["sequence"].tolist()
        X = aac_dpc(seqs) if self.meta["selected_feature"] == "AAC+DPC" else np.concatenate([aac_dpc(seqs), binary_profiles(seqs)], axis=1)
        return self.model.predict_proba(X)[:, 1]


# ================================================================ HemoPred
class HemoPredStrict:
    """Win 2017: RF over AAC+DPC; tree-count tuned by 10-fold CV on train
    (exact grid paywalled -> inferred; logged as deviation)."""

    GRID = {"n_estimators": [100, 300, 500, 800, 1000], "max_depth": [None, 10, 30, 50]}

    def train(self, df, gpu=-1, seed=42):
        X = aac_dpc(df["sequence"].tolist())
        y = df["label"].to_numpy(int)
        grid = GridSearchCV(
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            self.GRID, cv=StratifiedKFold(10, shuffle=True, random_state=seed),
            scoring="accuracy", n_jobs=-1,
        )
        grid.fit(X, y)
        self.model = grid.best_estimator_
        self.meta = {
            "citation": "Win et al., Future Med Chem 9:275 (2017)",
            "protocol": "RF over linear AAC+DPC composition; hyperparameters via 10-fold CV grid, selection = accuracy",
            "best_params": grid.best_params_, "cv_acc": grid.best_score_,
            "threshold_rule": "0.5",
            "deviations": "exact grid paywalled -> inferred grid {n_estimators,max_depth}; WEKA->sklearn RF",
        }
        return {"threshold": 0.5, "meta": self.meta}

    def predict_proba(self, df):
        return self.model.predict_proba(aac_dpc(df["sequence"].tolist()))[:, 1]


# ================================================================ HemoNet
class HemoNetStrict:
    """Yaseen 2021: Adam 1e-3 wd 1e-4, MSE(+/-1), early stop on train loss,
    Platt scaling on train predictions, 10-run probability averaging."""

    def train(self, df, gpu=0, seed=42):
        import torch
        from code.toxin_bench.methods.hemonet import Net, features

        dev = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")
        X = features(df).astype(np.float32)
        y01 = df["label"].to_numpy(int)
        ypm = np.where(y01 == 1, 1.0, -1.0).astype(np.float32)
        mu, sd = X.mean(0), X.std(0) + 1e-8
        Xn = torch.from_numpy((X - mu) / sd)
        probs_runs = []
        for run in range(10):
            torch.manual_seed(seed + run)
            model = Net(X.shape[1]).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.95)
            best_state, best_loss, patience, stale = None, np.inf, 10, 0
            for epoch in range(150):
                model.train()
                perm = torch.randperm(len(Xn))
                for i in range(0, len(Xn), 64):
                    idx = perm[i : i + 64]
                    loss = torch.nn.functional.mse_loss(model(Xn[idx].to(dev)), torch.from_numpy(ypm[idx]).to(dev))
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
                tr = float(torch.nn.functional.mse_loss(model(Xn.to(dev)), torch.from_numpy(ypm).to(dev)))
                if tr < best_loss - 1e-4:
                    best_loss, stale = tr, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= patience:
                        break
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                s = model(Xn.to(dev)).cpu().numpy()
            probs_runs.append(1 / (1 + np.exp(-s)))
        # Platt on averaged train scores
        scores = np.mean([np.log(p / (1 - p)) for p in probs_runs], axis=0)
        platt = LogisticRegression(max_iter=1000).fit(scores.reshape(-1, 1), y01)
        self.art = {"mu": mu, "sd": sd, "runs": [], "platt": platt, "dev": dev}
        for run in range(10):
            torch.manual_seed(seed + run)
            model = Net(X.shape[1]).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.95)
            best_state, best_loss, patience, stale = None, np.inf, 10, 0
            for epoch in range(150):
                model.train()
                perm = torch.randperm(len(Xn))
                for i in range(0, len(Xn), 64):
                    idx = perm[i : i + 64]
                    loss = torch.nn.functional.mse_loss(model(Xn[idx].to(dev)), torch.from_numpy(ypm[idx]).to(dev))
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
                tr = float(torch.nn.functional.mse_loss(model(Xn.to(dev)), torch.from_numpy(ypm).to(dev)))
                if tr < best_loss - 1e-4:
                    best_loss, stale = tr, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= patience:
                        break
            model.load_state_dict(best_state)
            self.art["runs"].append(model)
        self.meta = {
            "citation": "Yaseen et al., J Bioinform Comput Biol 19:2150021 (2021)",
            "protocol": "Adam 1e-3/wd1e-4, MSE on +/-1, early stop on train loss (patience 10), StepLR(50,0.95), 10 runs averaged, Platt scaling on train",
            "threshold_rule": "0.5 on Platt-calibrated probability",
            "deviations": "batch 64 / 150 max epochs (paper unspecified); per-paper Methods otherwise",
        }
        return {"threshold": 0.5, "meta": self.meta}

    def predict_proba(self, df):
        import torch
        from code.toxin_bench.methods.hemonet import features

        X = features(df).astype(np.float32)
        Xn = torch.from_numpy((X - self.art["mu"]) / self.art["sd"])
        dev = self.art["dev"]
        scores = []
        with torch.no_grad():
            for m in self.art["runs"]:
                m.eval()
                s = []
                for i in range(0, len(Xn), 256):
                    s.append(m(Xn[i : i + 256].to(dev)).cpu().numpy())
                scores.append(np.concatenate(s))
        score = np.mean(scores, axis=0)
        return self.art["platt"].predict_proba(score.reshape(-1, 1))[:, 1]


# ================================================================ LysePred
class LysePredStrict:
    """lincubator/LysePred arch; paper trains fixed epochs and keeps the
    best-TEST-ACC checkpoint. Leakage guard: select epoch on a 10% train-carved
    validation (MCC), 2 runs averaged. Everything else per repo defaults."""

    def train(self, df, gpu=0, seed=42):
        import torch
        from torch.utils.data import DataLoader
        from code.toxin_bench.methods.lysepred import Net, _DS, encode
        from code.toxin_filter.metrics import classification_metrics
        from sklearn.metrics import matthews_corrcoef

        dev = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")
        seqs = df["sequence"].tolist()
        y = df["label"].to_numpy(int)
        max_len = max(len(s) for s in seqs) + 1
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(seqs))
        n_val = max(1, int(0.1 * len(seqs)))
        va_idx, tr_idx = perm[:n_val], perm[n_val:]
        X = encode(seqs, max_len)
        run_probs = []
        for run in range(2):
            torch.manual_seed(seed + 1 + run)
            model = Net().to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0025)
            sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
            tr_loader = DataLoader(_DS(X[tr_idx], y[tr_idx]), batch_size=32, shuffle=True)
            best_state, best_mcc = None, -2.0
            for epoch in range(100):
                model.train()
                for xb, yb in tr_loader:
                    xb, yb = xb.to(dev), yb.to(dev)
                    loss = torch.nn.functional.cross_entropy(model(xb), yb)
                    loss = (loss - 0.02).abs() + 0.02
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
                model.eval()
                with torch.no_grad():
                    loader = DataLoader(_DS(X[va_idx], y[va_idx]), batch_size=256)
                    p = np.concatenate([torch.nn.functional.softmax(model(xb.to(dev)), 1)[:, 1].cpu().numpy() for xb, _ in loader])
                mcc = matthews_corrcoef(y[va_idx], (p >= 0.5).astype(int)) if len(np.unique(y[va_idx])) > 1 else -1
                if mcc > best_mcc:
                    best_mcc, best_state = mcc, {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            model.eval()
            self._last = model
            run_probs.append((model, best_mcc))
        self.art = {"runs": [m for m, _ in run_probs], "max_len": max_len, "dev": dev}
        self.meta = {
            "citation": "Lin et al., ACS Synth Biol (2026), LysePred",
            "protocol": "repo defaults (Adam 1e-4 wd2.5e-3, batch32, 100 epochs, CE+flooding, cosine restarts), 2 runs averaged",
            "threshold_rule": "0.5 on mean softmax",
            "deviations": "upstream selects best epoch on TEST accuracy -> switched to 10% train-carved validation MCC (no-leakage guard); 2 runs (repo train_repeat.sh uses 2)",
            "val_mcc_runs": [float(m) for _, m in run_probs],
        }
        return {"threshold": 0.5, "meta": self.meta}

    def predict_proba(self, df):
        import torch
        from torch.utils.data import DataLoader
        from code.toxin_bench.methods.lysepred import _DS, encode

        X = encode(df["sequence"].tolist(), self.art["max_len"])
        dev = self.art["dev"]
        probs = []
        with torch.no_grad():
            for m in self.art["runs"]:
                m.eval()
                loader = DataLoader(_DS(X, np.zeros(len(X), int)), batch_size=256)
                probs.append(np.concatenate([torch.nn.functional.softmax(m(xb.to(dev)), 1)[:, 1].cpu().numpy() for xb, _ in loader]))
        return np.mean(probs, axis=0)


# ================================================================ HemoPI2-RF
class HemoPI2RFStrict:
    """Rathore 2025: stratified 5-fold grid search (selection = mean accuracy),
    train-fold standardization, refit on full train; threshold 0.45 (ML models)."""

    GRID = {"n_estimators": [100, 200, 300, 400, 500], "max_depth": [None, 10, 20, 30],
            "max_features": ["sqrt", "log2"]}

    def train(self, df, gpu=-1, seed=42):
        from code.toxin_bench.methods.hemopi2 import composition_features

        X = composition_features(df["sequence"].tolist())
        y = df["label"].to_numpy(int)
        grid = GridSearchCV(
            RandomForestClassifier(random_state=seed, n_jobs=-1), self.GRID,
            cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            scoring="accuracy", n_jobs=-1,
        )
        grid.fit(X, y)
        self.model = grid.best_estimator_
        self.meta = {
            "citation": "Rathore et al., Commun Biol 8:176 (2025), HemoPI2 (ML arm)",
            "protocol": "stratified 5-fold CV grid search (mean accuracy), refit on full train; 1190 composition features (40aa truncation)",
            "best_params": grid.best_params_, "cv_acc": grid.best_score_,
            "threshold_rule": "0.45 (released ML threshold)",
            "deviations": "grid ranges inferred from paper ('various hyperparameter values'); no train standardization needed for RF",
        }
        return {"threshold": 0.45, "meta": self.meta}

    def predict_proba(self, df):
        from code.toxin_bench.methods.hemopi2 import composition_features
        return self.model.predict_proba(composition_features(df["sequence"].tolist()))[:, 1]


METHODS = {
    "hemopi1": HemoPI1Strict,
    "hemopred": HemoPredStrict,
    "hemonet": HemoNetStrict,
    "lysepred": LysePredStrict,
    "hemopi2_rf": HemoPI2RFStrict,
}
