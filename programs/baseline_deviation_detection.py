import argparse
import glob
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             confusion_matrix, roc_auc_score, average_precision_score)
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

RNG = np.random.default_rng(42)

HEATMAP_RE = re.compile(r'^mouse_heatmap_r(\d+)_c(\d+)$')

MOUSE_TIMING_COLS = [
    'mouse_avg_speed',
    'mouse_click_pause_mean',
    'mouse_click_pause_std',
    'mouse_left_hold_mean',
    'mouse_left_hold_std',
    'mouse_right_hold_mean',
    'mouse_right_hold_std'
]

KEYSTROKE_COLS = [
    'key_press_count',
    'key_avg_hold',
    'key_std_hold',
    'key_avg_dd',
    'key_std_dd',
    'key_avg_rp',
    'key_std_rp',
    'key_avg_rr',
    'key_cpm'
]

GUI_COLS = [
    'gui_focus_time',
    'gui_switch_count',
    'gui_unique_apps',
    'gui_window_event_count'
]

TIME_COLS = ['window_start_s', 'window_end_s']

CANON_MODAL_ORDER = ["mouse_heatmap", "mouse_timing", "keystrokes", "gui"]

def find_heatmap_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if HEATMAP_RE.match(c)]
    def key(c):
        m = HEATMAP_RE.match(c)
        return (int(m.group(1)), int(m.group(2)))
    return sorted(cols, key=key)

def load_csvs(file_patterns: List[str]) -> pd.DataFrame:
    files: List[str] = []
    for pat in file_patterns:
        files.extend(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"No files matched: {file_patterns}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        missing_time = [c for c in TIME_COLS if c not in df.columns]
        if missing_time:
            raise ValueError(f"{f} missing time columns: {missing_time}")
        heat_cols = find_heatmap_cols(df)
        missing_mouse_timing = [c for c in MOUSE_TIMING_COLS if c not in df.columns]
        missing_key = [c for c in KEYSTROKE_COLS if c not in df.columns]
        missing_gui = [c for c in GUI_COLS if c not in df.columns]
        if len(heat_cols) == 0:
            raise ValueError(f"{f} has no mouse heatmap columns.")
        if missing_mouse_timing:
            raise ValueError(f"{f} missing mouse timing cols: {missing_mouse_timing}")
        if missing_key:
            raise ValueError(f"{f} missing keystroke cols: {missing_key}")
        if missing_gui:
            raise ValueError(f"{f} missing GUI cols: {missing_gui}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


@dataclass
class ModalityConfig:
    name: str
    columns: List[str]
    use_pca: bool = False
    pca_var: float = 0.95
    scaler: str = "standard"

class IdentityTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

def make_preprocess_pipeline(cfg: ModalityConfig, n_features: int) -> Pipeline:
    steps: List[Tuple[str, TransformerMixin]] = []
    if cfg.scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif cfg.scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    else:
        steps.append(("scaler", IdentityTransformer()))
    if cfg.use_pca:
        steps.append(("pca", PCA(n_components=cfg.pca_var, svd_solver="full")))
    return Pipeline(steps)


def decision_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return np.ravel(s)
    elif hasattr(model, "score_samples"):
        return np.ravel(model.score_samples(X))
    else:
        yp = model.predict(X)
        return yp.astype(float)

def model_family_name(estimator: BaseEstimator) -> str:
    if isinstance(estimator, IsolationForest):
        return "IF"
    if isinstance(estimator, OneClassSVM):
        return "OCSVM"
    if isinstance(estimator, LocalOutlierFactor):
        return "LOF"
    if isinstance(estimator, EllipticEnvelope):
        return "EE"
    return estimator.__class__.__name__

@dataclass
class FittedModality:
    name: str
    pipeline: Pipeline
    model: BaseEstimator
    params: Dict
    threshold: float
    score_mean: float
    score_std: float

def _iforest_grid(alpha: float) -> Dict:
    return {
        "model": [IsolationForest(random_state=42, n_jobs=-1)],
        "model__n_estimators": [200, 400],
        "model__max_features": [0.6, 0.8, 1.0],
        "model__bootstrap": [False],
        "model__contamination": [alpha],
    }

def _ocsvm_grid(alpha: float) -> Dict:
    nus = sorted(set([alpha, min(0.05, max(alpha*2, 0.01)), 0.1]))
    return {
        "model": [OneClassSVM()],
        "model__kernel": ["rbf"],
        "model__nu": nus,
        "model__gamma": ["scale", "auto"],
    }

def _elliptic_grid(alpha: float) -> Dict:
    return {
        "model": [EllipticEnvelope(random_state=42)],
        "model__contamination": [alpha],
        "model__support_fraction": [0.7, 0.9],
    }

def _lof_grid(alpha: float) -> Dict:
    return {
        "model": [LocalOutlierFactor(novelty=True)],
        "model__n_neighbors": [20, 35, 50],
        "model__contamination": [alpha],
        "model__metric": ["minkowski"]
    }

def build_param_grid(alpha: float, families: Optional[List[str]] = None) -> List[Dict]:
    fams = set([f.upper() for f in families]) if families else {"IF","OCSVM","EE","LOF"}
    grids = []
    if "IF" in fams: grids.extend(list(ParameterGrid(_iforest_grid(alpha))))
    if "OCSVM" in fams: grids.extend(list(ParameterGrid(_ocsvm_grid(alpha))))
    if "EE" in fams: grids.extend(list(ParameterGrid(_elliptic_grid(alpha))))
    if "LOF" in fams: grids.extend(list(ParameterGrid(_lof_grid(alpha))))
    return grids

def unsupervised_objective(train_scores: np.ndarray, val_scores: Optional[np.ndarray] = None) -> float:
    mean_tr = float(np.mean(train_scores))
    std_tr = float(np.std(train_scores) + 1e-9)
    if val_scores is None or len(val_scores) == 0:
        return mean_tr - 0.5 * std_tr
    mean_val = float(np.mean(val_scores))
    std_val = float(np.std(val_scores) + 1e-9)
    return 0.5*(mean_tr + mean_val) - 0.25*(std_tr + std_val)

def choose_best_model(X_train: np.ndarray,
                      alpha: float,
                      X_impostor: Optional[np.ndarray] = None,
                      X_val: Optional[np.ndarray] = None,
                      families: Optional[List[str]] = None) -> Tuple[BaseEstimator, Dict, float]:
    best = None
    best_params = None
    best_score = -np.inf
    for params in build_param_grid(alpha, families=families):
        model = params["model"].set_params(**{k.replace("model__", ""): v for k, v in params.items() if k.startswith("model__")})
        try:
            model.fit(X_train)
            tr_scores = decision_scores(model, X_train)
            if X_impostor is not None and len(X_impostor) > 0:
                imp_scores = decision_scores(model, X_impostor)
                y = np.concatenate([np.ones_like(tr_scores), np.zeros_like(imp_scores)])
                s = np.concatenate([tr_scores, imp_scores])
                metric = roc_auc_score(y, s)
            else:
                val_scores = decision_scores(model, X_val) if X_val is not None else None
                metric = unsupervised_objective(tr_scores, val_scores)
        except Exception:
            continue
        if metric > best_score:
            best = model; best_params = params; best_score = metric
    if best is None:
        raise RuntimeError("Failed to fit any model; check data.")
    return best, best_params, best_score


class ModalityDetector:
    def __init__(self, cfg: ModalityConfig, alpha: float = 0.02, families: Optional[List[str]] = None):
        self.cfg = cfg
        self.alpha = alpha
        self.preprocess: Optional[Pipeline] = None
        self.model: Optional[BaseEstimator] = None
        self.threshold: Optional[float] = None
        self.score_mean: Optional[float] = None
        self.score_std: Optional[float] = None
        self.family: Optional[str] = None
        self.best_params: Optional[Dict] = None
        self.families = families

    def fit(self, X: np.ndarray, X_impostor: Optional[np.ndarray] = None, X_val: Optional[np.ndarray] = None):
        self.preprocess = make_preprocess_pipeline(self.cfg, X.shape[1])
        Z = self.preprocess.fit_transform(X)
        Z_imp = self.preprocess.transform(X_impostor) if X_impostor is not None else None
        Z_val = self.preprocess.transform(X_val) if X_val is not None else None
        model, params, _ = choose_best_model(Z, alpha=self.alpha, X_impostor=Z_imp, X_val=Z_val,
                                             families=self.families)
        self.model = model
        self.best_params = params
        self.family = model_family_name(model)
        tr_scores = decision_scores(self.model, Z)
        q = np.quantile(tr_scores, self.alpha)
        self.threshold = float(q)
        self.score_mean = float(np.mean(tr_scores))
        self.score_std = float(np.std(tr_scores) + 1e-9)

    def scores(self, X: np.ndarray) -> np.ndarray:
        if self.preprocess is None or self.model is None:
            raise RuntimeError("ModalityDetector not fitted.")
        Z = self.preprocess.transform(X)
        return decision_scores(self.model, Z)

    def anomaly_prob(self, X: np.ndarray) -> np.ndarray:
        s = self.scores(X)
        z = (s - self.score_mean) / max(self.score_std, 1e-9)
        z_th = (self.threshold - self.score_mean) / max(self.score_std, 1e-9)
        p = 1.0 / (1.0 + np.exp((z - z_th)))
        return p

    def to_dict(self) -> Dict:
        return {
            "cfg": asdict(self.cfg),
            "alpha": self.alpha,
            "threshold": self.threshold,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "family": self.family,
            "best_params": self.best_params
        }

@dataclass
class EnsembleConfig:
    alpha: float = 0.02
    use_logit_meta: bool = True

class MultimodalEnsemble:
    def __init__(self, modality_cfgs: List[ModalityConfig], ens_cfg: EnsembleConfig,
                 families: Optional[List[str]] = None,
                 family_map: Optional[Dict[str, List[str]]] = None):
        self.ens_cfg = ens_cfg
        self.meta: Optional[LogisticRegression] = None
        self.modality_names = [cfg.name for cfg in modality_cfgs]
        self.modalities = []
        fams_global = [f.upper() for f in families] if families is not None else None
        for cfg in modality_cfgs:
            fams_for_mod = None
            if family_map and cfg.name in family_map:
                fams_for_mod = [f.upper() for f in family_map[cfg.name]]
            elif fams_global is not None:
                fams_for_mod = fams_global
            self.modalities.append(ModalityDetector(cfg, alpha=ens_cfg.alpha, families=fams_for_mod))

    def fit(self, X_train_by_mod: Dict[str, np.ndarray],
            X_imp_by_mod: Optional[Dict[str, np.ndarray]] = None,
            X_val_by_mod: Optional[Dict[str, np.ndarray]] = None):
        for m in self.modalities:
            X_tr = X_train_by_mod[m.cfg.name]
            X_imp = X_imp_by_mod.get(m.cfg.name) if X_imp_by_mod else None
            X_val = X_val_by_mod.get(m.cfg.name) if X_val_by_mod else None
            m.fit(X_tr, X_imp, X_val)

        if X_imp_by_mod and self.ens_cfg.use_logit_meta:
            S_tr = []
            S_imp = []
            for m in self.modalities:
                S_tr.append(m.scores(X_train_by_mod[m.cfg.name]))
                S_imp.append(m.scores(X_imp_by_mod[m.cfg.name]))
            S_tr = np.vstack(S_tr).T
            S_imp = np.vstack(S_imp).T
            X_meta = np.vstack([S_tr, S_imp])
            y_meta = np.hstack([np.zeros(len(S_tr)), np.ones(len(S_imp))])  # 1 = anomaly
            self.meta = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
            self.meta.fit(X_meta, y_meta)

    def anomaly_scores(self, X_by_mod: Dict[str, np.ndarray]) -> pd.DataFrame:
        probs = {}
        raw_scores = {}
        for m in self.modalities:
            probs[m.cfg.name] = m.anomaly_prob(X_by_mod[m.cfg.name])
            raw_scores[m.cfg.name] = m.scores(X_by_mod[m.cfg.name])
        df_probs = pd.DataFrame(probs)
        df_raw = pd.DataFrame({f"{k}_raw": v for k, v in raw_scores.items()})
        if self.meta is not None:
            X_meta = df_raw[[c for c in df_raw.columns if c.endswith("_raw")]].values
            final = self.meta.predict_proba(X_meta)[:, 1]
        else:
            final = df_probs.mean(axis=1).values
        out = pd.concat([df_probs, df_raw], axis=1)
        out["anomaly_score"] = final
        return out

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        payload = {"ens_cfg": asdict(self.ens_cfg), "modality_names": self.modality_names}
        joblib.dump(self, os.path.join(path, "model.joblib"))
        with open(os.path.join(path, "model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def load(path: str) -> "MultimodalEnsemble":
        return joblib.load(os.path.join(path, "model.joblib"))

def split_modalities(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    heat_cols = find_heatmap_cols(df)
    X_heat = df[heat_cols].values.astype(float)
    X_mouse_t = df[MOUSE_TIMING_COLS].values.astype(float)
    X_keys = df[KEYSTROKE_COLS].values.astype(float)
    X_gui = df[GUI_COLS].values.astype(float)
    return {"mouse_heatmap": X_heat, "mouse_timing": X_mouse_t, "keystrokes": X_keys, "gui": X_gui}

def build_default_modality_cfgs(df_sample: pd.DataFrame) -> List[ModalityConfig]:
    heat_cols = find_heatmap_cols(df_sample)
    cfgs = [
        ModalityConfig(name="mouse_heatmap", columns=heat_cols, use_pca=True, pca_var=0.95, scaler="standard"),
        ModalityConfig(name="mouse_timing", columns=MOUSE_TIMING_COLS, use_pca=False, scaler="robust"),
        ModalityConfig(name="keystrokes", columns=KEYSTROKE_COLS, use_pca=False, scaler="robust"),
        ModalityConfig(name="gui", columns=GUI_COLS, use_pca=False, scaler="robust"),
    ]
    return cfgs

def cmd_train(args):
    df_tr = load_csvs(args.train)
    df_imp = load_csvs(args.impostor) if args.impostor else None
    modality_cfgs = build_default_modality_cfgs(df_tr.head(1))
    X_tr_by = split_modalities(df_tr)
    X_imp_by = split_modalities(df_imp) if df_imp is not None else None
    ens_cfg = EnsembleConfig(alpha=args.alpha, use_logit_meta=(df_imp is not None))
    model = MultimodalEnsemble(modality_cfgs, ens_cfg)
    model.fit(X_tr_by, X_imp_by, None)
    model.save(args.out)
    if df_imp is not None:
        S_tr = model.anomaly_scores(X_tr_by)["anomaly_score"].values
        S_imp = model.anomaly_scores(X_imp_by)["anomaly_score"].values
        y = np.concatenate([np.zeros_like(S_tr), np.ones_like(S_imp)])
        s = np.concatenate([S_tr, S_imp])
        auc = roc_auc_score(y, s)
        th = np.quantile(S_tr, 0.99)
        fpr = float(np.mean(S_tr >= th))
        tpr = float(np.mean(S_imp >= th))
        print(f"[TRAIN] Meta AUC (baseline vs impostor): {auc:.4f}")
        print(f"[TRAIN] Example threshold @99th quantile -> FPR={fpr:.4f}, TPR={tpr:.4f}")
    print(f"Model saved to: {args.out}")

def cmd_score(args):
    model = MultimodalEnsemble.load(args.model)
    df_in = load_csvs(args.input)
    X_by = split_modalities(df_in)
    scores = model.anomaly_scores(X_by)
    out = pd.concat([df_in[TIME_COLS].reset_index(drop=True), scores], axis=1)
    out.to_csv(args.output, index=False)
    print(f"Scores saved to: {args.output}")

def ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if 'window_start_s' in df.columns:
        return df.sort_values('window_start_s', kind="stable").reset_index(drop=True)
    return df.reset_index(drop=True)

def plot_curve(x, y, xlabel: str, ylabel: str, title: str, path: str):
    try:
        plt.figure()
        plt.plot(x, y)
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
        plt.grid(True); plt.tight_layout()
        plt.savefig(path, dpi=160)
    finally:
        plt.close()

def _probs_matrix(model: MultimodalEnsemble, X_by: Dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    mats = []
    for m in model.modalities:
        p = m.anomaly_prob(X_by[m.cfg.name][idx])
        mats.append(p)
    return np.vstack(mats).T  # (n, 4)

def _single_modality_table(model: MultimodalEnsemble, X_by: Dict[str, np.ndarray],
                           train_idx, val_idx, test_norm_idx, test_anom_idx,
                           quantiles: List[float]) -> pd.DataFrame:
    rows = []
    y_test = np.concatenate([np.zeros(len(test_norm_idx), dtype=int),
                             np.ones(len(test_anom_idx), dtype=int)])
    for m in model.modalities:
        def p_on(idx):
            return m.anomaly_prob(X_by[m.cfg.name][idx])
        cal = np.concatenate([p_on(train_idx), p_on(val_idx)])
        test = np.concatenate([p_on(test_norm_idx), p_on(test_anom_idx)])
        aauc = roc_auc_score(y_test, test)
        ap = average_precision_score(y_test, test)
        for q in quantiles:
            tau = float(np.quantile(cal, q))
            y_pred = (test >= tau).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
            acc = (tp+tn)/len(y_test)
            prec = tp/max(tp+fp,1)
            rec = tp/max(tp+fn,1)
            f1 = 2*prec*rec/max(prec+rec,1e-9)
            rows.append({
                "modality": m.cfg.name,
                "chosen_family": m.family,
                "q": q, "tau": tau,
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "roc_auc": aauc, "avg_precision": ap,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
            })
    return pd.DataFrame(rows)

def fusion_score_matrix(M: np.ndarray, kind: str, val_probs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, str]:
    k = kind.lower()
    if k == "mean":
        return M.mean(axis=1), ""
    if k == "max":
        return M.max(axis=1), ""
    if k == "weighted":
        if val_probs is None or val_probs.size == 0:
            w = np.ones(M.shape[1]) / M.shape[1]
        else:
            v = np.var(val_probs, axis=0) + 1e-6
            w = (1.0 / v); w = w / w.sum()
        return M.dot(w), f"w={w.tolist()}"
    if k == "kof4":
        return (M >= 0.5).mean(axis=1), ""
    raise ValueError(f"Unknown fusion kind: {kind}")

def _fusion_table(model: MultimodalEnsemble, X_by: Dict[str, np.ndarray],
                  train_idx, val_idx, test_norm_idx, test_anom_idx,
                  quantiles: List[float]) -> pd.DataFrame:
    rows = []
    y_test = np.concatenate([np.zeros(len(test_norm_idx), dtype=int),
                             np.ones(len(test_anom_idx), dtype=int)])
    cal_idx = np.concatenate([train_idx, val_idx])
    Mcal = _probs_matrix(model, X_by, cal_idx)
    Mval = _probs_matrix(model, X_by, val_idx)
    Mtest = _probs_matrix(model, X_by, np.concatenate([test_norm_idx, test_anom_idx]))

    def eval_scores(scores, tau, q):
        y_pred = (scores >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
        acc = (tp+tn)/len(y_test)
        prec = tp/max(tp+fp,1)
        rec = tp/max(tp+fn,1)
        f1 = 2*prec*rec/max(prec+rec,1e-9)
        return {"q": q, "tau": tau, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    for q in quantiles:
        tau_mean = float(np.quantile(Mcal.mean(axis=1), q))
        tau_max  = float(np.quantile(Mcal.max(axis=1),  q))
        w = 1.0/(np.var(Mval, axis=0)+1e-6); w = w / w.sum()
        tau_weighted = float(np.quantile(Mcal.dot(w), q))
        tau_kof4 = float(np.quantile((Mcal>=0.5).mean(axis=1), q))

        rows.append({"fusion":"mean", **eval_scores(Mtest.mean(axis=1), tau_mean, q)})
        rows.append({"fusion":"max",  **eval_scores(Mtest.max(axis=1),  tau_max,  q)})
        rows.append({"fusion":"weighted", **eval_scores(Mtest.dot(w),   tau_weighted, q)})
        rows.append({"fusion":"kof4", **eval_scores((Mtest>=0.5).mean(axis=1), tau_kof4, q)})

    return pd.DataFrame(rows)

def _fit_ensemble_for_combo(df_sample: pd.DataFrame, alpha: float,
                            family_map: Optional[Dict[str, List[str]]],
                            train_idx, val_idx) -> MultimodalEnsemble:
    modality_cfgs = build_default_modality_cfgs(df_sample.head(1))
    ens_cfg = EnsembleConfig(alpha=alpha, use_logit_meta=False)
    model = MultimodalEnsemble(modality_cfgs, ens_cfg,
                               families=None,
                               family_map=family_map)
    X_by = split_modalities(df_sample)
    X_tr_by = {k: v[train_idx] for k, v in X_by.items()}
    X_val_by = {k: v[val_idx] for k, v in X_by.items()}
    model.fit(X_tr_by, X_imp_by_mod=None, X_val_by_mod=X_val_by)
    return model

def _algo_family_table(df_sample: pd.DataFrame, alpha: float, families_to_compare: List[str],
                       X_by: Dict[str, np.ndarray], train_idx, val_idx, test_norm_idx, test_anom_idx,
                       quantiles: List[float]) -> pd.DataFrame:
    rows = []
    y_test = np.concatenate([np.zeros(len(test_norm_idx), dtype=int),
                             np.ones(len(test_anom_idx), dtype=int)])
    cal_idx = np.concatenate([train_idx, val_idx])
    for fam in [f.upper() for f in families_to_compare]:
        fam_map = {m: [fam] for m in CANON_MODAL_ORDER}
        model_fam = _fit_ensemble_for_combo(df_sample, alpha, fam_map, train_idx, val_idx)
        Mcal = _probs_matrix(model_fam, X_by, cal_idx)
        Mtest = _probs_matrix(model_fam, X_by, np.concatenate([test_norm_idx, test_anom_idx]))
        scores_mean = Mtest.mean(axis=1)
        for q in quantiles:
            tau = float(np.quantile(Mcal.mean(axis=1), q))
            y_pred = (scores_mean >= tau).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
            acc = (tp+tn)/len(y_test)
            prec = tp/max(tp+fp,1)
            rec = tp/max(tp+fn,1)
            f1 = 2*prec*rec/max(prec+rec,1e-9)
            rows.append({"family": fam, "fusion": "mean", "q": q, "tau": tau,
                         "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                         "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return pd.DataFrame(rows)

def _save_tables(out_dir: str, single_df: pd.DataFrame, fusion_df: pd.DataFrame, algo_df: pd.DataFrame):
    tbl_dir = os.path.join(out_dir, "tables")
    os.makedirs(tbl_dir, exist_ok=True)
    p_single = os.path.join(tbl_dir, "table_single_modality.csv")
    p_fusion = os.path.join(tbl_dir, "table_fusion_strategies.csv")
    p_algo   = os.path.join(tbl_dir, "table_algorithm_families.csv")
    single_df.to_csv(p_single, index=False)
    fusion_df.to_csv(p_fusion, index=False)
    algo_df.to_csv(p_algo, index=False)
    try:
        xlsx = os.path.join(tbl_dir, "tables.xlsx")
        with pd.ExcelWriter(xlsx) as w:
            single_df.to_excel(w, sheet_name="single_modality", index=False)
            fusion_df.to_excel(w, sheet_name="fusion_strategies", index=False)
            algo_df.to_excel(w, sheet_name="algorithm_families", index=False)
        print(f"[TABLES] Saved Excel: {xlsx}")
    except Exception as e:
        print(f"[TABLES] Skipped Excel export ({e}).")
    print(f"[TABLES] Saved:\n  {p_single}\n  {p_fusion}\n  {p_algo}")

def cmd_pair_experiment(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df_u1 = load_csvs(args.user1)
    df_u2 = load_csvs(args.user2)
    df_u1 = ensure_sorted(df_u1)
    df_u2 = ensure_sorted(df_u2)

    n1 = min(len(df_u1), args.n_per_user)
    n2 = min(len(df_u2), args.n_per_user)
    if n1 < args.n_per_user or n2 < args.n_per_user:
        print(f"[WARN] Not enough rows; using n1={n1}, n2={n2}")
    df_u1 = df_u1.iloc[:n1].reset_index(drop=True)
    df_u2 = df_u2.iloc[:n2].reset_index(drop=True)

    n_keep_u2 = min(args.n_user2_keep, len(df_u2))
    df_u2_keep = df_u2.iloc[:n_keep_u2].reset_index(drop=True)

    data = pd.concat([df_u1, df_u2_keep], ignore_index=True)
    train_n = args.train_n; val_n = args.val_n
    assert train_n + val_n <= len(df_u1), "Train+Val exceeds user1 windows"

    train_idx = np.arange(0, train_n)
    val_idx = np.arange(train_n, train_n + val_n)
    test_norm_idx = np.arange(train_n + val_n, n1)
    test_anom_idx = np.arange(n1, n1 + n_keep_u2)

    modality_cfgs = build_default_modality_cfgs(data.head(1))
    ens_cfg = EnsembleConfig(alpha=args.alpha, use_logit_meta=False)
    model = MultimodalEnsemble(modality_cfgs, ens_cfg, families=None, family_map=None)
    X_by = split_modalities(data)
    X_tr_by = {k: v[train_idx] for k, v in X_by.items()}
    X_val_by = {k: v[val_idx] for k, v in X_by.items()}
    model.fit(X_tr_by, X_imp_by_mod=None, X_val_by_mod=X_val_by)

    X_all_by = split_modalities(data)
    cal_idx = np.concatenate([train_idx, val_idx])
    S_cal = model.anomaly_scores({k: v[cal_idx] for k, v in X_all_by.items()})["anomaly_score"].values
    final_th = float(np.quantile(S_cal, args.final_thresh_q))

    S_norm = model.anomaly_scores({k: v[test_norm_idx] for k, v in X_all_by.items()})["anomaly_score"].values
    S_anom = model.anomaly_scores({k: v[test_anom_idx] for k, v in X_all_by.items()})["anomaly_score"].values

    y_true = np.concatenate([np.zeros_like(S_norm), np.ones_like(S_anom)])
    scores = np.concatenate([S_norm, S_anom])
    y_pred = (scores >= final_th).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    acc = float((tp + tn) / max(len(y_true), 1))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-9))
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)

    metrics = {
        "threshold": final_th,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": float(auc), "avg_precision": float(ap),
        "counts": {"normal": int(len(S_norm)), "anomaly": int(len(S_anom))}
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fpr, tpr, _ = roc_curve(y_true, scores)
    pr_prec, pr_rec, _ = precision_recall_curve(y_true, scores)
    plot_curve(fpr, tpr, "FPR", "TPR", "ROC Curve", os.path.join(args.out_dir, "roc_curve.png"))
    plot_curve(pr_rec, pr_prec, "Recall", "Precision", "PR Curve", os.path.join(args.out_dir, "pr_curve.png"))

    df_times = data.loc[np.concatenate([test_norm_idx, test_anom_idx]), TIME_COLS].reset_index(drop=True)
    df_scores = model.anomaly_scores({k: v[np.concatenate([test_norm_idx, test_anom_idx])] for k, v in X_all_by.items()}).reset_index(drop=True)
    df_scores["label"] = y_true; df_scores["pred"] = y_pred
    out_csv = os.path.join(args.out_dir, "test_scores.csv")
    pd.concat([df_times, df_scores], axis=1).to_csv(out_csv, index=False)

    quantiles = [float(q) for q in (args.table_quantiles or [0.99])]
    fams_to_compare = [f.upper() for f in (args.table_families or ["IF","OCSVM","LOF","EE"])]
    single_df = _single_modality_table(model, X_all_by, train_idx, val_idx, test_norm_idx, test_anom_idx, quantiles)
    fusion_df = _fusion_table(model, X_all_by, train_idx, val_idx, test_norm_idx, test_anom_idx, quantiles)
    algo_df = _algo_family_table(data, args.alpha, fams_to_compare, X_all_by, train_idx, val_idx, test_norm_idx, test_anom_idx, quantiles)
    _save_tables(args.out_dir, single_df, fusion_df, algo_df)

    print(json.dumps(metrics, indent=2))
    print(f"[OUT] Saved to {args.out_dir}")

def parse_combo_spec(spec: str) -> Optional[Dict[str, List[str]]]:
    """
    Returns a family_map {modality_name: [families]} or None for AUTO.
    Syntax (case-insensitive):
      - "auto"                         -> None (search all families per modality)
      - "all:IF"                       -> every modality restricted to IF
      - "each:IF,EE,EE,EE"             -> in order [mouse_heatmap, mouse_timing, keystrokes, gui]
      - "map:mouse_heatmap=IF,gui=EE"  -> only listed modalities restricted; others AUTO
    Families allowed: IF, OCSVM, LOF, EE
    """
    s = spec.strip().lower()
    if s == "auto":
        return None
    if s.startswith("all:"):
        fam = s.split(":", 1)[1].strip().upper()
        return {m: [fam] for m in CANON_MODAL_ORDER}
    if s.startswith("each:"):
        parts = s.split(":", 1)[1].split(",")
        parts = [p.strip().upper() for p in parts]
        if len(parts) != 4:
            raise ValueError("each: expects 4 families in modality order: mouse_heatmap, mouse_timing, keystrokes, gui")
        return {m: [fam] for m, fam in zip(CANON_MODAL_ORDER, parts)}
    if s.startswith("map:"):
        body = s.split(":", 1)[1]
        fammap: Dict[str, List[str]] = {}
        for kv in body.split(","):
            kv = kv.strip()
            if not kv:
                continue
            k, v = kv.split("=")
            fammap[k.strip()] = [v.strip().upper()]
        return fammap
    raise ValueError(f"Unrecognized combo spec: {spec}")

def _take_user_block(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = ensure_sorted(df)
    n = min(n, len(df))
    return df.iloc[:n].reset_index(drop=True)

def _index_splits(n1: int, keep2: int, train_n: int, val_n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_idx = np.arange(0, train_n)
    val_idx = np.arange(train_n, train_n + val_n)
    test_norm_idx = np.arange(train_n + val_n, n1)
    test_anom_idx = np.arange(n1, n1 + keep2)
    return train_idx, val_idx, test_norm_idx, test_anom_idx

def _fit_ensemble_for_combo(data_concat: pd.DataFrame, alpha: float,
                            family_map: Optional[Dict[str, List[str]]],
                            train_idx, val_idx) -> MultimodalEnsemble:
    modality_cfgs = build_default_modality_cfgs(data_concat.head(1))
    ens_cfg = EnsembleConfig(alpha=alpha, use_logit_meta=False)
    model = MultimodalEnsemble(modality_cfgs, ens_cfg,
                               families=None,
                               family_map=family_map)
    X_by = split_modalities(data_concat)
    X_tr_by = {k: v[train_idx] for k, v in X_by.items()}
    X_val_by = {k: v[val_idx] for k, v in X_by.items()}
    model.fit(X_tr_by, X_imp_by_mod=None, X_val_by_mod=X_val_by)
    return model

def _evaluate_pair_once(base_name: str, imp_name: str,
                        df_base: pd.DataFrame, df_imp: pd.DataFrame,
                        combos: List[str], fusions: List[str],
                        n_per_user: int, keep_u2: int,
                        train_n: int, val_n: int,
                        alpha: float, q: float) -> List[Dict]:
    u1 = _take_user_block(df_base, n_per_user)
    u2 = _take_user_block(df_imp,  n_per_user)
    u2_keep = u2.iloc[:min(keep_u2, len(u2))].reset_index(drop=True)
    data = pd.concat([u1, u2_keep], ignore_index=True)

    n1 = len(u1); keep2 = len(u2_keep)
    train_idx, val_idx, test_norm_idx, test_anom_idx = _index_splits(n1, keep2, train_n, val_n)

    X_by = split_modalities(data)
    cal_idx = np.concatenate([train_idx, val_idx])
    test_idx = np.concatenate([test_norm_idx, test_anom_idx])
    y_test = np.concatenate([np.zeros(len(test_norm_idx), dtype=int),
                             np.ones(len(test_anom_idx), dtype=int)])

    rows = []
    for combo in combos:
        fam_map = parse_combo_spec(combo)
        model = _fit_ensemble_for_combo(data, alpha, fam_map, train_idx, val_idx)

        chosen = []
        name_map = {m.cfg.name: m.family for m in model.modalities}
        for mname in CANON_MODAL_ORDER:
            chosen.append(name_map.get(mname, "NA"))
        modalities_str = ",".join(chosen)

        Mcal = np.column_stack([m.anomaly_prob(X_by[m.cfg.name][cal_idx]) for m in model.modalities])
        Mval = np.column_stack([m.anomaly_prob(X_by[m.cfg.name][val_idx]) for m in model.modalities])
        Mtest = np.column_stack([m.anomaly_prob(X_by[m.cfg.name][test_idx]) for m in model.modalities])

        for fkind in fusions:
            def fusion_scores(M, kind, val_probs=None):
                k = kind.lower()
                if k == "mean":      return M.mean(axis=1)
                if k == "max":       return M.max(axis=1)
                if k == "weighted":
                    if val_probs is None or val_probs.size == 0:
                        w = np.ones(M.shape[1]) / M.shape[1]
                    else:
                        v = np.var(val_probs, axis=0) + 1e-6
                        w = (1.0 / v); w = w / w.sum()
                    return M.dot(w)
                if k == "kof4":      return (M >= 0.5).mean(axis=1)
                raise ValueError(f"Unknown fusion kind: {kind}")
            scores_cal = fusion_scores(Mcal, fkind, val_probs=Mval)
            scores_test = fusion_scores(Mtest, fkind, val_probs=Mval)
            tau = float(np.quantile(scores_cal, q))
            y_pred = (scores_test >= tau).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
            acc = (tp + tn) / len(y_test)
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-9)

            rows.append({
                "base_user": base_name,
                "imp_user": imp_name,
                "direction": f"{base_name}->{imp_name}",
                "modalities": modalities_str,
                "combo_spec": combo,
                "fusion": fkind,
                "q": q,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
            })
    return rows

def cmd_pair_grid(args):
    data_dir = Path(args.data_dir)
    pat = args.pattern or "user*.csv"
    files = sorted(data_dir.glob(pat))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {data_dir} with pattern {pat}")

    users: Dict[str, pd.DataFrame] = {}
    for f in files:
        name = f.stem
        users[name] = load_csvs([str(f)])

    def _sort_key(s: str):
        m = re.search(r'(\d+)$', s)
        return (int(m.group(1)) if m else 10**9, s)
    names = sorted(users.keys(), key=_sort_key)

    if args.undirected:
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i+1, len(names))]
    else:
        pairs = [(a, b) for a in names for b in names if a != b]

    combos = args.combos or ["auto", "all:IF", "all:OCSVM", "all:LOF", "all:EE"]
    fusions = [f.lower() for f in (args.fusions or ["mean"])]

    all_rows: List[Dict] = []
    for a, b in pairs:
        rows = _evaluate_pair_once(
            base_name=a, imp_name=b,
            df_base=users[a], df_imp=users[b],
            combos=combos, fusions=fusions,
            n_per_user=args.n_per_user, keep_u2=args.n_user2_keep,
            train_n=args.train_n, val_n=args.val_n,
            alpha=args.alpha, q=args.final_thresh_q
        )
        all_rows.extend(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(all_rows)
    out_csv = out_dir / "pair_grid_results.csv"
    df_out.to_csv(out_csv, index=False)

    if args.summary:
        cols = ["base_user","imp_user","modalities","fusion","q","tn","fp","fn","tp","accuracy","precision","recall","f1"]
        df_out[cols].to_csv(out_dir / "pair_grid_summary.csv", index=False)

    print(f"[GRID] Wrote {len(df_out)} rows to {out_csv}")
    if args.summary:
        print(f"[GRID] Wrote summary to {out_dir / 'pair_grid_summary.csv'}")

def build_arg_parser():
    p = argparse.ArgumentParser(description="Multimodal Baseline Deviation Detection")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train per-modality models + ensemble")
    p_train.add_argument("--train", nargs="+", required=True, help="Glob(s) for baseline CSV files (one user).")
    p_train.add_argument("--impostor", nargs="+", help="Optional glob(s) for impostor/anomaly CSV files.")
    p_train.add_argument("--alpha", type=float, default=0.02, help="Per-modality quantile for threshold (default=0.02).")
    p_train.add_argument("--out", required=True, help="Output model directory.")
    p_train.set_defaults(func=cmd_train)

    p_score = sub.add_parser("score", help="Score new windows with a trained model")
    p_score.add_argument("--model", required=True, help="Path to model directory (created by 'train').")
    p_score.add_argument("--input", nargs="+", required=True, help="Glob(s) for CSV file(s) to score.")
    p_score.add_argument("--output", required=True, help="Output CSV path for scores.")
    p_score.set_defaults(func=cmd_score)

    p_pair = sub.add_parser("pair_experiment", help="Run the 40/10/10+30 split (user1 vs user2), report metrics, and write tables")
    p_pair.add_argument("--user1", nargs="+", required=True, help="Glob(s) for user1 CSV (>=60 rows recommended).")
    p_pair.add_argument("--user2", nargs="+", required=True, help="Glob(s) for user2 CSV (>=60 rows recommended).")
    p_pair.add_argument("--n_per_user", type=int, default=60, help="Windows to take from each user (default 60).")
    p_pair.add_argument("--n_user2_keep", type=int, default=30, help="How many user2 windows to keep for anomaly test (default 30).")
    p_pair.add_argument("--train_n", type=int, default=40, help="User1 windows for training (default 40).")
    p_pair.add_argument("--val_n", type=int, default=10, help="User1 windows for hyperparam tuning (default 10).")
    p_pair.add_argument("--alpha", type=float, default=0.02, help="Per-modality quantile for thresholds.")
    p_pair.add_argument("--final_thresh_q", type=float, default=0.99, help="Quantile on baseline final scores (default 0.99).")
    p_pair.add_argument("--out_dir", required=True, help="Directory to save metrics/curves/scores/tables.")
    p_pair.add_argument("--model_out", help="Optional: save trained model directory.")
    p_pair.add_argument("--load_model", help="Optional: load an existing model instead of training.")
    p_pair.add_argument("--table_quantiles", nargs="*", type=float, default=[0.95, 0.97, 0.99],
                        help="Quantiles to evaluate (e.g., 0.95 0.97 0.99).")
    p_pair.add_argument("--table_families", nargs="*", default=["IF", "OCSVM", "LOF", "EE"],
                        help="Algorithm families to compare in Table C.")
    p_pair.set_defaults(func=cmd_pair_experiment)

    p_grid = sub.add_parser("pair_grid", help="Run all user pairs from a folder with modality/fusion combinations; write one results CSV")
    p_grid.add_argument("--data_dir", default="data", help="Folder with user CSVs named like user1.csv, user2.csv, ...")
    p_grid.add_argument("--pattern", default="user*.csv", help="Glob pattern inside data_dir")
    p_grid.add_argument("--undirected", action="store_true", help="If set, run each unordered pair once (a,b) with a as baseline. Otherwise run both directions.")
    p_grid.add_argument("--n_per_user", type=int, default=60)
    p_grid.add_argument("--n_user2_keep", type=int, default=30)
    p_grid.add_argument("--train_n", type=int, default=40)
    p_grid.add_argument("--val_n", type=int, default=10)
    p_grid.add_argument("--alpha", type=float, default=0.02, help="Per-modality quantile for thresholds/calibration")
    p_grid.add_argument("--final_thresh_q", type=float, default=0.99, help="Calibration quantile on fused baseline scores")
    p_grid.add_argument("--out_dir", required=True, help="Output directory for the final CSV table(s)")
    p_grid.add_argument("--combos", nargs="*", default=["auto","all:IF","all:OCSVM","all:LOF","all:EE"],
                        help=("Per-modality detector specifications. Examples: "
                              "'auto' | 'all:IF' | 'each:IF,EE,EE,EE' | "
                              "'map:mouse_heatmap=IF,mouse_timing=EE,keystrokes=EE,gui=EE'"))
    p_grid.add_argument("--fusions", nargs="*", default=["mean","max","weighted","kof4"],
                        help="Fusion functions to evaluate: mean, max, weighted, kof4")
    p_grid.add_argument("--summary", action="store_true", help="Also write a compact summary CSV")
    p_grid.set_defaults(func=cmd_pair_grid)

    return p

def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
