#!/usr/bin/env python3
"""
kernel_diagnostics_nestedcv.py

Nested-CV kernel diagnostics pipeline using project loader and kernel builder.

See in-script configuration for defaults (R repeats, outer_folds=5, inner Optuna trials=50, B=500 perms).
Outputs:
 - detail_{dataset}__{kernel}.csv  (one row per outer-fold per repeat per kernel)
 - summary_by_dataset_kernel.csv   (aggregated medians + IQR per dataset×kernel)
 - pairwise_tests_{dataset}.csv    (pairwise Wilcoxon + median-delta + CI + FDR per metric family)
 - eigenvalues/{dataset}__{kernel}__eig.npy
 - run_metadata.json
"""
import argparse
import json
import logging
import time
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import trustworthiness
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.svm import SVC

import optuna
import warnings
warnings.filterwarnings("ignore")

# Try to import project functions (must be in PYTHONPATH)
try:
    from Data.loader import load_all_datasets
except Exception as e:
    raise ImportError("Could not import Data.loader.load_all_datasets. Ensure your repo is on PYTHONPATH.") from e

try:
    from Kernels.all_kernels import build_single_kernel
except Exception as e:
    raise ImportError("Could not import Kernels.all_kernels.build_single_kernel. Ensure your repo is on PYTHONPATH.") from e

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("kernel_diag_nestedcv")

# ----------------- Configuration / defaults -----------------
DEFAULT_OUT_DIR = Path("kernel_diag_nestedcv_out")
DEFAULT_R = 5               # repeats of outer CV
DEFAULT_OUTER_FOLDS = 2
DEFAULT_INNER_TRIALS = 5      # budget per kernel per outer fold
DEFAULT_B_PERMS = 100
DEFAULT_N_EIGS_SAVE = 100
DEFAULT_LAMBDA_DOF = 1e-3
DEFAULT_RANDOM_SEED = 2025
DEFAULT_TRUST_K = 5
# metrics to compute and later to compare
DIAGNOSTIC_METRICS = [
    "cka", "cka_p",
    "hsic", "hsic_p",
    "effective_rank", "top1_ratio", "cum_top3", "cum_top5",
    "kpca_top1", "kpca_cum_top5",
    f"knn_overlap_{5}", f"knn_overlap_{10}",
    "fisher_ratio_rkhs",
    "svm_mean_margin", "svm_std_margin", "svm_sv_prop", "svm_n_sv",
    "dof_trace"
]

# ----------------- Helpers -----------------
def ensure_symmetric(K):
    K = np.asarray(K, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("Kernel matrix must be square.")
    return (K + K.T) / 2.0

def center_kernel(K):
    n = K.shape[0]
    one = np.ones((n, n), dtype=float) / n
    return K - one @ K - K @ one + one @ K @ one

def make_label_matrix(y):
    # robust one-hot for multiple sklearn versions
    y = np.asarray(y).reshape(-1, 1)
    try:
        from sklearn.preprocessing import OneHotEncoder
        try:
            enc = OneHotEncoder(sparse=False)
        except TypeError:
            enc = OneHotEncoder(sparse_output=False)
        Y = enc.fit_transform(y)
        if hasattr(Y, "toarray"):
            Y = Y.toarray()
        return np.asarray(Y, dtype=float)
    except Exception:
        # fallback: manual one-hot
        labels, inv = np.unique(y, return_inverse=True)
        Y = np.zeros((len(y), labels.size), dtype=float)
        Y[np.arange(len(y)), inv] = 1.0
        return Y

def kernel_distance_matrix_from_K(K):
    diag = np.diag(K)
    D2 = diag[:, None] + diag[None, :] - 2.0 * K
    D2 = np.maximum(D2, 0.0)
    return np.sqrt(D2)

def cka_from_grams(K, L):
    Kc = center_kernel(K)
    Lc = center_kernel(L)
    num = float(np.trace(Kc.T @ Lc))
    den = np.linalg.norm(Kc, 'fro') * np.linalg.norm(Lc, 'fro')
    return np.nan if den == 0 else num / den

def hsic_empirical(K, L):
    n = K.shape[0]
    if n <= 1:
        return np.nan
    Kc = center_kernel(K)
    Lc = center_kernel(L)
    return float(np.trace(Kc @ Lc) / ((n - 1) ** 2))

def perm_test(func, K, L, n_permutations=500, seed=0):
    rng = np.random.RandomState(seed)
    obs = func(K, L)
    n = K.shape[0]
    if np.isnan(obs):
        return np.nan, np.nan
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        Lp = L[perm][:, perm]
        val = func(K, Lp)
        if not np.isnan(val) and val >= obs:
            count += 1
    p = (count + 1) / (n_permutations + 1)
    return float(p), float(obs)

def kernel_eigenspectrum(K, top_n=200):
    Kc = center_kernel(K)
    vals, vecs = eigh(Kc)
    vals = np.maximum(vals, 0.0)[::-1]
    total = vals.sum() if vals.sum() > 0 else 1.0
    explained = vals / total
    cum_expl = np.cumsum(explained)
    s_sum = vals.sum()
    erank = float((s_sum ** 2) / np.sum(vals ** 2)) if s_sum > 0 and np.sum(vals ** 2) > 0 else 0.0
    return {'eigvals': vals[:top_n], 'full': vals, 'explained': explained, 'cum_expl': cum_expl, 'erank': erank}

def knn_overlap(X, K, k_list=(5,10)):
    n = X.shape[0]
    if n <= 1:
        return {f"knn_overlap_{k}": np.nan for k in k_list}
    orig_d = squareform(pdist(X, metric='euclidean'))
    kernel_d = kernel_distance_matrix_from_K(K)
    out = {}
    for k in k_list:
        if k >= n:
            out[f"knn_overlap_{k}"] = np.nan
            continue
        overlaps = []
        for i in range(n):
            idx_orig = np.argsort(orig_d[i])[1:k+1]   # exclude self
            idx_k = np.argsort(kernel_d[i])[1:k+1]
            overlaps.append(len(set(idx_orig).intersection(idx_k)) / float(k))
        out[f"knn_overlap_{k}"] = float(np.mean(overlaps))
    return out

def fisher_ratio_rkhs(K, y):
    n = K.shape[0]
    if n <= 1:
        return np.nan
    D = kernel_distance_matrix_from_K(K)
    y = np.asarray(y)
    within = []
    between = []
    for i in range(n):
        for j in range(i+1, n):
            if y[i] == y[j]:
                within.append(D[i, j])
            else:
                between.append(D[i, j])
    if len(within) == 0 or len(between) == 0:
        return np.nan
    return float(np.mean(between) / (np.mean(within) if np.mean(within) != 0 else np.nan))

def svm_margin_stats_precomputed(K_train, K_test, y_train, y_test, C=1.0):
    # precomputed kernel SVC: fit with K_train, predict with K_test (shape: n_test x n_train)
    try:
        le = LabelEncoder()
        ytr = le.fit_transform(y_train)
        yte = le.transform(y_test)
        if len(np.unique(ytr)) != 2:
            return {'mean_margin': np.nan, 'std_margin': np.nan, 'sv_prop': np.nan, 'n_sv': np.nan}
        model = SVC(C=C, kernel='precomputed')
        model.fit(K_train, ytr)
        # decision_function for training samples (we can compute on K_train)
        margins_train = model.decision_function(K_train)
        mean_m = float(np.mean(np.abs(margins_train)))
        std_m = float(np.std(margins_train))
        n_sv = int(np.sum(model.n_support_))
        sv_prop = float(n_sv / K_train.shape[0])
        return {'mean_margin': mean_m, 'std_margin': std_m, 'sv_prop': sv_prop, 'n_sv': n_sv}
    except Exception:
        return {'mean_margin': np.nan, 'std_margin': np.nan, 'sv_prop': np.nan, 'n_sv': np.nan}

def degrees_of_freedom(K, lam=1e-3):
    n = K.shape[0]
    try:
        inv = np.linalg.inv(K + lam * np.eye(n))
        tr = float(np.trace(K @ inv))
        return tr
    except Exception:
        return np.nan

# ----------------- kernel builder / gram utility -----------------
def sanitize_kernel_params(params):
    kp = dict(params) if params else {}
    if 'gamma' in kp:
        try:
            if isinstance(kp['gamma'], str) and kp['gamma'] not in ['auto', 'scale']:
                kp['gamma'] = float(kp['gamma'])
            elif not isinstance(kp['gamma'], str):
                kp['gamma'] = float(kp['gamma'])
        except Exception:
            kp['gamma'] = kp.get('gamma', 'scale')
    return kp

def get_kernel_function_from_builder(kernel_name, params):
    kp = sanitize_kernel_params(params)
    info = build_single_kernel(kernel_name, **kp)
    if not info or 'func' not in info:
        raise RuntimeError(f"build_single_kernel returned no callable for {kernel_name}")
    return info['func']

def compute_gram_from_callable(kfunc, X, X2=None):
    # try kfunc(X, X) signature, else kfunc(X) returning Gram
    try:
        if X2 is None:
            K = kfunc(X, X)
        else:
            K = kfunc(X, X2)
        return np.asarray(K, dtype=float)
    except TypeError:
        try:
            K = kfunc(X)
            return np.asarray(K, dtype=float)
        except Exception:
            raise

# ----------------- Inner optimization (Optuna) objective -----------------
def make_inner_evaluator(X_train, y_train, kernel_name, param_suggestions, inner_cv_splits=3, seed=0):
    """
    Return objective(trial) that Optuna will use. param_suggestions is a function
    that given an optuna trial returns params dict for the kernel + C.
    The evaluation uses StratifiedKFold(inner_cv_splits) on (X_train,y_train).
    """
    def objective(trial):
        params = param_suggestions(trial)
        # obtain kernel function
        try:
            kfunc = get_kernel_function_from_builder(kernel_name, params)
        except Exception:
            # return poor score so optimizer avoids this
            return 0.0
        cv = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=seed)
        accs = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            Xt, Xv = X_train[tr_idx], X_train[val_idx]
            yt, yv = y_train[tr_idx], y_train[val_idx]
            # remove constant columns based on training fold
            var_mask = Xt.var(axis=0) != 0
            if var_mask.sum() == 0:
                return 0.0
            Xt_r = Xt[:, var_mask]
            Xv_r = Xv[:, var_mask]
            try:
                K_tt = compute_gram_from_callable(kfunc, Xt_r, Xt_r)
                K_vt = compute_gram_from_callable(kfunc, Xv_r, Xt_r)
            except Exception:
                return 0.0
            # train precomputed SVM on K_tt
            try:
                C = params.get('C', 1.0)
                model = SVC(C=C, kernel='precomputed')
                model.fit(K_tt, yt)
                ypred = model.predict(K_vt)
                accs.append(np.mean(ypred == yv))
            except Exception:
                return 0.0
        return float(np.mean(accs)) if accs else 0.0
    return objective

# default param suggestion replicating your earlier script's search ranges
def default_param_suggestions(kernel_name):
    def sugg(trial):
        params = {}
        params['C'] = trial.suggest_float('C', 0.001, 100.0, log=True)
        if kernel_name == 'poly':
            params['degree'] = trial.suggest_int('degree', 1, 6)
            params['coef0'] = 1.0
            params['gamma'] = trial.suggest_float('gamma', 2**-6, 2**2, log=True)
        elif kernel_name == 'rbf':
            params['gamma'] = trial.suggest_float('gamma', 2**-6, 2**2, log=True)
        elif kernel_name == 'custom_hermite':
            params['degree'] = trial.suggest_int('degree', 1, 6)
        elif kernel_name == 'custom_gegen':
            params['degree'] = trial.suggest_int('degree', 1, 6)
            params['alpha'] = trial.suggest_float('alpha', -0.49, 1.5)
        elif kernel_name == 'custom_alsalam':
            params['degree'] = trial.suggest_int('degree', 1, 6)
            params['a'] = trial.suggest_categorical('a', [-1])
            params['q'] = trial.suggest_float('q', 0.01, 0.99)
        return params
    return sugg

# ----------------- bootstrap helpers -----------------
def bootstrap_median_delta_ci(deltas, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.RandomState(seed)
    deltas = np.asarray(deltas)
    deltas = deltas[~np.isnan(deltas)]
    if deltas.size == 0:
        return np.nan, np.nan
    med = np.median(deltas)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(deltas, size=deltas.size, replace=True)
        boots.append(np.median(sample))
    lower = np.percentile(boots, 100*alpha/2)
    upper = np.percentile(boots, 100*(1-alpha/2))
    return float(lower), float(upper)

def benjamini_hochberg(p_values):
    p = np.asarray(p_values)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    cummin = 1.0
    for i in range(n-1, -1, -1):
        q = ranked[i] * n / (i+1)
        cummin = min(cummin, q)
        adjusted[i] = min(cummin, 1.0)
    adj = np.empty(n, dtype=float)
    adj[order] = adjusted
    return adj

# ----------------- Main pipeline -----------------
def analyze(config_path: str,
            kernels: list,
            output_dir: str,
            R=DEFAULT_R,
            outer_folds=DEFAULT_OUTER_FOLDS,
            n_trials_inner=DEFAULT_INNER_TRIALS,
            n_permutations=DEFAULT_B_PERMS,
            n_eigs_save=DEFAULT_N_EIGS_SAVE,
            lambda_dof=DEFAULT_LAMBDA_DOF,
            trust_k=DEFAULT_TRUST_K,
            random_seed=DEFAULT_RANDOM_SEED):
    t0 = time.time()
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    eigdir = outdir / "eigenvalues"; eigdir.mkdir(exist_ok=True)
    metadata = {
        "R": R, "outer_folds": outer_folds, "n_trials_inner": n_trials_inner,
        "n_permutations": n_permutations, "n_eigs_save": n_eigs_save,
        "lambda_dof": lambda_dof, "trust_k": trust_k, "random_seed": random_seed,
        "kernels": kernels
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    with open(config_path, 'r') as f:
        cfg = json.load(f)
    data_cfg = cfg['data']
    datasets = load_all_datasets(
        uci_list=data_cfg.get('uci', []),
        libsvm_dir=Path(data_cfg.get('libsvm_dir', "")),
        libsvm_files=data_cfg.get('libsvm', [])
    )
    logger.info(f"Loaded {len(datasets)} datasets. Running diagnostics...")

    # store per-dataset rows (one row per repeat×outer_fold×kernel)
    all_rows = []
    rng_master = np.random.RandomState(random_seed)

    for dataset_name, X_raw, y in datasets:
        logger.info(f"Dataset: {dataset_name} (n={X_raw.shape[0]}, d={X_raw.shape[1]})")
        # scale to [-1, 1] globally first (we will also remove constant cols per fold)
        scaler = MinMaxScaler(feature_range=(-1,1))
        X_all = scaler.fit_transform(X_raw)
        n = X_all.shape[0]
        # label matrix full (we will subset per test fold)
        Y_onehot_full = make_label_matrix(y)

        # We'll run R repeats: for each repeat create an outer StratifiedKFold with a different seed
        for repeat_idx in range(R):
            outer_seed = int(rng_master.randint(0, 2**31-1))
            skf_outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=outer_seed)
            fold_idx = 0
            for train_idx, test_idx in skf_outer.split(X_all, y):
                fold_seed = int(rng_master.randint(0, 2**31-1))
                X_train_raw = X_all[train_idx]
                y_train = np.asarray(y)[train_idx]
                X_test_raw = X_all[test_idx]
                y_test = np.asarray(y)[test_idx]

                # remove constant columns computed on training fold
                var_mask = X_train_raw.var(axis=0) != 0
                if var_mask.sum() == 0:
                    logger.warning(f"{dataset_name} rep{repeat_idx} fold{fold_idx}: all features constant after var mask, skipping fold.")
                    fold_idx += 1
                    continue
                X_train = X_train_raw[:, var_mask]
                X_test = X_test_raw[:, var_mask]
                Y_test_onehot = Y_onehot_full[test_idx][:, :]  # subset

                # prepare L = Y_test Y_test^T for test fold
                L_test = Y_test_onehot @ Y_test_onehot.T

                # For pairing later ensure we compute metrics for all kernels on the same fold
                kernel_fold_values = {}

                for kernel_name in kernels:
                    logger.info(f"  Dataset {dataset_name} rep{repeat_idx} fold{fold_idx} kernel {kernel_name}")

                    # Inner optimization on training fold to select hyperparams
                    param_sugg = default_param_suggestions(kernel_name)
                    objective = make_inner_evaluator(X_train, y_train, kernel_name, param_sugg, inner_cv_splits=3, seed=fold_seed+1)
                    sampler = optuna.samplers.TPESampler(seed=fold_seed+2)
                    study = optuna.create_study(direction='maximize', sampler=sampler, study_name=f"{dataset_name}_{kernel_name}_rep{repeat_idx}_fold{fold_idx}")
                    try:
                        study.optimize(objective, n_trials=n_trials_inner, show_progress_bar=False)
                        best_params = study.best_params if study.best_trial else {}
                    except Exception as e:
                        logger.warning(f"    Optuna failed for {kernel_name}: {e}")
                        best_params = {}

                    # Build kernel function with best params
                    try:
                        kfunc = get_kernel_function_from_builder(kernel_name, best_params)
                    except Exception as e:
                        logger.warning(f"    build_single_kernel failed for {kernel_name} with params {best_params}: {e}")
                        # fallback to standard kernels for basic cases
                        kfunc = None

                    # compute gram matrices for train/train, test/test, test/train
                    try:
                        if kfunc is not None:
                            K_tt = ensure_symmetric(compute_gram_from_callable(kfunc, X_train, X_train))
                            K_te = compute_gram_from_callable(kfunc, X_test, X_train)  # test x train
                            K_ee = ensure_symmetric(compute_gram_from_callable(kfunc, X_test, X_test))
                        else:
                            # fallback kernels for robustness
                            from sklearn.metrics.pairwise import pairwise_kernels
                            if kernel_name == 'linear':
                                K_tt = ensure_symmetric(X_train @ X_train.T)
                                K_te = X_test @ X_train.T
                                K_ee = ensure_symmetric(X_test @ X_test.T)
                            elif kernel_name == 'rbf':
                                # median heuristic on training distances
                                sq = squareform(pdist(X_train, 'sqeuclidean')) if X_train.shape[0] > 1 else np.array([[0.0]])
                                med = np.median(sq[np.triu_indices(X_train.shape[0], k=1)]) if X_train.shape[0] > 1 else 1.0
                                gamma = 1.0/(2.0*med) if med>0 else 1.0
                                K_tt = pairwise_kernels(X_train, X_train, metric='rbf', gamma=gamma)
                                K_te = pairwise_kernels(X_test, X_train, metric='rbf', gamma=gamma)
                                K_ee = pairwise_kernels(X_test, X_test, metric='rbf', gamma=gamma)
                            elif kernel_name == 'poly':
                                K_tt = pairwise_kernels(X_train, X_train, metric='polynomial', degree=3, gamma=1.0, coef0=1.0)
                                K_te = pairwise_kernels(X_test, X_train, metric='polynomial', degree=3, gamma=1.0, coef0=1.0)
                                K_ee = pairwise_kernels(X_test, X_test, metric='polynomial', degree=3, gamma=1.0, coef0=1.0)
                            else:
                                logger.warning(f"    No Gram available for {kernel_name}, skipping kernel for this fold.")
                                continue
                    except Exception as e:
                        logger.error(f"    Gram computation failed for {dataset_name}/{kernel_name}: {e}")
                        continue

                    # CENTER K_ee and use it for CKA/HSIC/spectrum/KPCA etc.
                    try:
                        K_ee_centered = center_kernel(K_ee)
                    except Exception:
                        K_ee_centered = K_ee.copy()

                    # Permutation tests for CKA and HSIC on test fold
                    try:
                        cka_p, cka_val = perm_test(cka_from_grams, K_ee, L_test, n_permutations=n_permutations, seed=fold_seed+10)
                    except Exception:
                        cka_p, cka_val = np.nan, cka_from_grams(K_ee, L_test)

                    try:
                        hsic_p, hsic_val = perm_test(hsic_empirical, K_ee, L_test, n_permutations=n_permutations, seed=fold_seed+20)
                    except Exception:
                        hsic_p, hsic_val = np.nan, hsic_empirical(K_ee, L_test)

                    # spectrum & KPCA
                    spec = kernel_eigenspectrum(K_ee, top_n=n_eigs_save)
                    erank = spec['erank']
                    full_vals = spec['full']
                    top1_ratio = float(full_vals[0] / full_vals.sum()) if full_vals.size>0 and full_vals.sum()>0 else np.nan
                    cum_top3 = float(spec['cum_expl'][min(2, spec['cum_expl'].size-1)]) if spec['cum_expl'].size>0 else np.nan
                    cum_top5 = float(spec['cum_expl'][min(4, spec['cum_expl'].size-1)]) if spec['cum_expl'].size>0 else np.nan
                    kpca_top1 = float(spec['explained'][0]) if spec['explained'].size>0 else np.nan
                    kpca_cum_top5 = cum_top5

                    # trustworthiness between original Euclidean on test and kernel-induced distances
                    try:
                        tw = trustworthiness(X_test, kernel_distance_matrix_from_K(K_ee), n_neighbors=trust_k)
                    except Exception:
                        # sklearn.trustworthiness expects arrays; compute with original and embedding distances via distance matrix workaround:
                        try:
                            # compute pairwise distances and use trustworthiness by passing 2 arrays; fallback uses sklearn implementation
                            tw = trustworthiness(X_test, X_test, n_neighbors=trust_k)
                        except Exception:
                            tw = np.nan

                    knn_res = knn_overlap(X_test, K_ee, k_list=(5,10))

                    # fisher ratio in RKHS
                    fisher = fisher_ratio_rkhs(K_ee, y_test)

                    # svm margins and n_support (train on K_tt precomputed and compute metrics on train)
                    svm_stats = svm_margin_stats_precomputed(K_tt, K_te, y_train, y_test, C=best_params.get('C', 1.0) if isinstance(best_params, dict) else 1.0)

                    # degrees of freedom
                    dof = degrees_of_freedom(K_ee, lam=lambda_dof)

                    # Save eigenvalues (full)
                    eig_file = eigdir / f"{dataset_name}__{kernel_name}__rep{repeat_idx}_fold{fold_idx}__eig.npy"
                    try:
                        np.save(eig_file, spec['full'])
                    except Exception:
                        pass

                    # collect row for this repeat/fold/kernel
                    row = {
                        "dataset": dataset_name, "repeat": int(repeat_idx), "fold": int(fold_idx),
                        "kernel": kernel_name, "n_samples_test": int(X_test.shape[0]),
                        "n_train": int(X_train.shape[0]),
                        "cka": float(cka_val), "cka_p": float(cka_p),
                        "hsic": float(hsic_val), "hsic_p": float(hsic_p),
                        "effective_rank": float(erank),
                        "top1_ratio": float(top1_ratio),
                        "cum_top3": float(cum_top3),
                        "cum_top5": float(cum_top5),
                        "kpca_top1": float(kpca_top1),
                        "kpca_cum_top5": float(kpca_cum_top5),
                        "fisher_ratio_rkhs": float(fisher),
                        "dof_trace": float(dof),
                        "svm_mean_margin": float(svm_stats.get('mean_margin', np.nan)),
                        "svm_std_margin": float(svm_stats.get('std_margin', np.nan)),
                        "svm_sv_prop": float(svm_stats.get('sv_prop', np.nan)),
                        "svm_n_sv": float(svm_stats.get('n_sv', np.nan))
                    }
                    row.update(knn_res)
                    # include best_params summary for traceability
                    row["best_params"] = json.dumps(best_params)
                    all_rows.append(row)
                    kernel_fold_values[kernel_name] = row

                # end kernels for this fold
                fold_idx += 1
            # end folds for this repeat
        # end repeats for this dataset

        # After dataset done, write per-dataset detail files grouped by kernel
        df_rows = pd.DataFrame([r for r in all_rows if r['dataset'] == dataset_name])
        # Save detail per kernel×repeat×fold rows
        for kernel in kernels:
            dfk = df_rows[df_rows['kernel'] == kernel]
            if not dfk.empty:
                out_file = outdir / f"detail_{dataset_name}__{kernel}.csv"
                dfk.to_csv(out_file, index=False)
        # save eigenvalues already saved individually

    # After all datasets finished, save summary and pairwise tests
    df_all = pd.DataFrame(all_rows)
    if df_all.empty:
        logger.error("No diagnostic rows produced. Exiting.")
        return
    summary_file = outdir / "detail_all_rows.csv"
    df_all.to_csv(summary_file, index=False)
    logger.info(f"Wrote detailed rows to {summary_file}")

    # produce aggregated summary_by_dataset_kernel (median + IQR across repeats/folds)
    agg_rows = []
    for (dataset_name, kernel_name), group in df_all.groupby(['dataset', 'kernel']):
        stats = {}
        stats['dataset'] = dataset_name
        stats['kernel'] = kernel_name
        for metric in DIAGNOSTIC_METRICS:
            if metric in group.columns:
                vals = group[metric].dropna().values
                stats[f"{metric}_median"] = float(np.median(vals)) if vals.size>0 else np.nan
                stats[f"{metric}_iqr"] = float(np.percentile(vals, 75)-np.percentile(vals,25)) if vals.size>0 else np.nan
            else:
                stats[f"{metric}_median"] = np.nan
                stats[f"{metric}_iqr"] = np.nan
        agg_rows.append(stats)
    df_summary = pd.DataFrame(agg_rows)
    df_summary.to_csv(outdir / "summary_by_dataset_kernel.csv", index=False)
    logger.info(f"Wrote summary_by_dataset_kernel.csv")

    # Pairwise kernel comparisons per dataset: Wilcoxon paired across repeat×fold matching
    # For each dataset and each metric, compare all kernel pairs over aligned samples.
    pairwise_records = []
    for dataset_name, df_ds in df_all.groupby('dataset'):
        # create index keys for pairing: (repeat, fold)
        df_ds = df_ds.set_index(['repeat','fold','kernel']).sort_index()
        # get list of distinct (repeat,fold) combinations that have all kernels present
        combos = df_ds.index.remove_unused_levels() if hasattr(df_ds.index, 'remove_unused_levels') else df_ds.index
        # easier: pivot table with index (repeat,fold) and columns kernels for metric
        for metric in DIAGNOSTIC_METRICS:
            # build pivot (repeat,fold) x kernel with metric values
            try:
                pivot = df_ds[metric].unstack(level='kernel')
            except Exception:
                pivot = pd.pivot_table(df_ds.reset_index(), index=['repeat','fold'], columns='kernel', values=metric)
            # drop rows with any NaN across the kernels to ensure pairing
            pivot_clean = pivot.dropna(how='any')
            kernels_present = pivot_clean.columns.tolist()
            # all pairwise combos
            for a,b in combinations(kernels_present, 2):
                vals_a = pivot_clean[a].values
                vals_b = pivot_clean[b].values
                if vals_a.size < 5:
                    # too few paired samples for Wilcoxon
                    w_stat, pval = np.nan, np.nan
                    median_delta = np.nan
                    ci_low, ci_high = np.nan, np.nan
                else:
                    try:
                        w = wilcoxon(vals_a, vals_b, zero_method='wilcox', alternative='two-sided')
                        w_stat, pval = float(w.statistic), float(w.pvalue)
                    except Exception:
                        w_stat, pval = np.nan, np.nan
                    deltas = vals_a - vals_b
                    median_delta = float(np.median(deltas))
                    ci_low, ci_high = bootstrap_median_delta_ci(deltas, n_boot=1000, seed=random_seed)
                pairwise_records.append({
                    "dataset": dataset_name,
                    "metric": metric,
                    "kernel_A": a, "kernel_B": b,
                    "n_paired": int(vals_a.size),
                    "median_delta": median_delta,
                    "delta_ci_low": ci_low, "delta_ci_high": ci_high,
                    "wilcoxon_stat": w_stat, "wilcoxon_p": pval
                })
    df_pairwise = pd.DataFrame(pairwise_records)
    # multiple testing correction per metric family (i.e. for each metric)
    adjusted_p = []
    for metric, sub in df_pairwise.groupby('metric'):
        pvals = sub['wilcoxon_p'].fillna(1.0).values
        adj = benjamini_hochberg(pvals)
        adjusted_p.extend(adj.tolist())
    if len(adjusted_p) == len(df_pairwise):
        df_pairwise['wilcoxon_p_fdr'] = adjusted_p
    df_pairwise.to_csv(outdir / "pairwise_tests_by_dataset_metric.csv", index=False)
    logger.info(f"Wrote pairwise tests to {outdir/'pairwise_tests_by_dataset_metric.csv'}")

    # Save metadata and finish
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(f"All done in {time.time()-t0:.1f}s. Outputs in {outdir}")

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Nested-CV Kernel Diagnostics (uses your project's loader + kernel builder)")
    p.add_argument('--config', type=str, required=True, help='Path to config.json used by your loader')
    p.add_argument('--kernels', nargs='+', required=True, help='List of kernel names to evaluate (strings used in build_single_kernel)')
    p.add_argument('--output_dir', type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument('--R', type=int, default=DEFAULT_R, help='Number of outer-CV repeats')
    p.add_argument('--outer_folds', type=int, default=DEFAULT_OUTER_FOLDS)
    p.add_argument('--n_trials_inner', type=int, default=DEFAULT_INNER_TRIALS, help='Optuna trials per kernel per fold')
    p.add_argument('--n_permutations', type=int, default=DEFAULT_B_PERMS)
    p.add_argument('--n_eigs_save', type=int, default=DEFAULT_N_EIGS_SAVE)
    p.add_argument('--lambda_dof', type=float, default=DEFAULT_LAMBDA_DOF)
    p.add_argument('--trust_k', type=int, default=DEFAULT_TRUST_K)
    p.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED)
    return p.parse_args()

def main():
    args = parse_args()
    analyze(config_path=args.config,
            kernels=args.kernels,
            output_dir=args.output_dir,
            R=args.R,
            outer_folds=args.outer_folds,
            n_trials_inner=args.n_trials_inner,
            n_permutations=args.n_permutations,
            n_eigs_save=args.n_eigs_save,
            lambda_dof=args.lambda_dof,
            trust_k=args.trust_k,
            random_seed=args.seed)

if __name__ == "__main__":
    main()
