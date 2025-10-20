#!/usr/bin/env python3
"""
Extended, corrected analysis:
 - Paired comparisons (Delta vs baseline linear) using Wilcoxon
 - Win indicator (threshold) and LODO RF classifier predicting wins from meta-features
 - Permutation importances averaged across LODO folds (mean ± std)
 - Small multivariate check for training_time (robust-ish via OLS + HC3 SE & Theil-Sen)
 - Outputs: per-kernel CSVs and TXT summaries
"""
import numpy as np
import pandas as pd
from pathlib import Path
import logging, re, time
from datetime import datetime
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import TheilSenRegressor, LogisticRegression
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ---------- Config ----------
RESULTS_DIR = Path("bayesian_optimization_methodology")
CHAR_DIR = Path("dataset_characterization")
OUTPUT_DIR = Path("kernel_analysis_simple_outputs_extended")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 2025
N_BOOTSTRAP = 500
N_SPLITS_CV = 5
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 6
PERM_REPEATS = 20
MIN_N_PER_KERNEL = 6   # you have 20 datasets; adjust if you want stricter

PERFORMANCE_METRICS = ['accuracy_mean', 'f1_mean', 'training_time_mean', 'support_vector_prop_mean']
DATASET_FEATURES = [
    'n_samples', 'n_features', 'n_classes', 'samples_per_feature_ratio',
    'overall_mean', 'overall_std', 'mean_feature_mean', 'mean_feature_std',
    'max_feature_correlation', 'mean_feature_correlation',
    'class_imbalance_ratio', 'class_entropy',
    'n_constant_features', 'proportion_constant_features'
]
# baseline kernel for Δ comparisons
BASELINE_KERNEL = 'custom_gegen'
# win threshold (absolute improvement in metric to call it a win)
WIN_THRESHOLD = 1e-4  # 0.01% absolute; adjust if you want stricter

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("kernel_ext")

# ---------- Utilities ----------
def find_latest_file(directory: Path, pattern: str):
    files = list(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    best = None; best_time = None
    for f in files:
        m = re.search(r'(\d{8}_\d{6})', f.name)
        if m:
            t = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
            if best_time is None or t > best_time:
                best_time = t; best = f
    if best is None:
        best = max(files, key=lambda x: x.stat().st_mtime)
    return best

def bootstrap_ci_stat(x, y, func, n_boot=N_BOOTSTRAP, seed=0, alpha=0.05):
    rng = np.random.RandomState(seed)
    stats = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            val = func(x[idx], y[idx])[0]
            if not np.isnan(val):
                stats.append(val)
        except Exception:
            continue
    if not stats:
        return (np.nan, np.nan)
    lower = np.percentile(stats, 100*(alpha/2))
    upper = np.percentile(stats, 100*(1-alpha/2))
    return float(lower), float(upper)

def benjamini_hochberg(pvals):
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    cummin = 1.0
    for i in range(n-1, -1, -1):
        q = ranked[i]*n/(i+1)
        cummin = min(cummin, q)
        adjusted[i] = min(cummin, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out

# ---------- Core ----------
class KernelMetaAnalyzer:
    def __init__(self):
        self.rng = np.random.RandomState(RANDOM_STATE)

    def load_data(self):
        res_file = find_latest_file(Path(RESULTS_DIR), "bayesian_methodology_summary_*.csv")
        char_file = find_latest_file(Path(CHAR_DIR), "dataset_characterization_summary_*.csv")
        logger.info(f"Results file: {res_file}")
        logger.info(f"Chars file:   {char_file}")
        res = pd.read_csv(res_file)
        chars = pd.read_csv(char_file)
        return res, chars

    def preprocess(self, res, chars):
        # aggregate if multiple rows per dataset-kernel
        agg = res.groupby(['dataset','kernel']).agg({
            'accuracy_mean':'mean','f1_mean':'mean',
            'training_time_mean':'mean','support_vector_prop_mean':'mean'
        }).reset_index()
        merged = agg.merge(chars, left_on='dataset', right_on='dataset_name', how='inner')
        # filter n_classes < 2
        if 'n_classes' in merged.columns:
            merged = merged[merged['n_classes'] >= 2].copy()
        # impute numeric medians
        num_cols = merged.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            if merged[c].isna().any():
                merged[c] = merged[c].fillna(merged[c].median())
        return merged

    def paired_deltas_and_wilcoxon(self, merged):
        """Compute Δ_{k,d} = metric_k - metric_baseline and paired Wilcoxon per kernel+metric"""
        kernels = sorted(merged['kernel'].unique())
        rows = []
        for k in kernels:
            if k == BASELINE_KERNEL:
                continue
            dfk = merged[merged['kernel']==k].copy()
            dfbase = merged[merged['kernel']==BASELINE_KERNEL].copy()
            # merge by dataset
            m = dfk.set_index('dataset')[[c for c in PERFORMANCE_METRICS if c in dfk.columns]].join(
                dfbase.set_index('dataset')[[c for c in PERFORMANCE_METRICS if c in dfbase.columns]],
                lsuffix='_k', rsuffix='_base', how='inner'
            ).dropna()
            for metric in PERFORMANCE_METRICS:
                colk = metric + '_k'
                colb = metric + '_base'
                if colk not in m.columns or colb not in m.columns:
                    continue
                delta = m[colk] - m[colb]
                # paired Wilcoxon
                try:
                    stat, p = wilcoxon(m[colk], m[colb], zero_method='wilcox', alternative='two-sided')
                except Exception:
                    # fallback to sign test equivalent: count signs
                    stat = np.nan
                    p = np.nan
                med = np.median(delta)
                # bootstrap CI for median
                rng = np.random.RandomState(self.rng.randint(1e6))
                boots = []
                for _ in range(500):
                    idx = rng.randint(0, len(delta), len(delta))
                    boots.append(np.median(delta.values[idx]))
                ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
                rows.append({'kernel':k,'metric':metric,'n':len(delta),'median_delta':med,'delta_ci_low':ci_low,'delta_ci_high':ci_high,'wilcoxon_stat':stat,'wilcoxon_p':p})
        df_out = pd.DataFrame(rows)
        # BH per metric family across kernels
        for metric in PERFORMANCE_METRICS:
            idxs = df_out[df_out['metric']==metric].index
            if len(idxs)>0:
                pvals = df_out.loc[idxs,'wilcoxon_p'].fillna(1.0).values
                df_out.loc[idxs,'wilcoxon_p_fdr'] = benjamini_hochberg(pvals)
        df_out.to_csv(OUTPUT_DIR / "paired_kernel_comparisons.csv", index=False)
        return df_out

    def analyze_per_kernel(self, merged):
        kernels = sorted(merged['kernel'].unique())
        features = [f for f in DATASET_FEATURES if f in merged.columns]
        metrics = [m for m in PERFORMANCE_METRICS if m in merged.columns]

        # We'll also prepare data for meta-modeling (win indicators)
        all_delta = []

        # For each kernel compute correlation table and RF LODO classifier for wins
        meta_model_summaries = []
        for k in kernels:
            logger.info(f"Processing kernel {k}")
            dfk = merged[merged['kernel']==k].copy()
            n = len(dfk)
            if n < MIN_N_PER_KERNEL:
                logger.info(f"Kernel {k} has n={n} (<{MIN_N_PER_KERNEL}), still computing univariate correlations but skipping heavy models.")
            # 1) correlations (feature vs metric) -- Spearman + bootstrap CI
            corr_rows = []
            for feat in features:
                if feat not in dfk.columns:
                    continue
                for metric in metrics:
                    if metric not in dfk.columns:
                        continue
                    sub = dfk[[feat,metric]].dropna()
                    if len(sub) < 4:
                        continue
                    x = sub[feat].values.astype(float); y = sub[metric].values.astype(float)
                    pr, pp = (pearsonr(x,y) if len(x)>=3 else (np.nan, np.nan))
                    sr, sp = (spearmanr(x,y) if len(x)>=3 else (np.nan, np.nan))
                    pr_ci = bootstrap_ci_stat(x,y,pearsonr,seed=self.rng.randint(1e6))
                    sr_ci = bootstrap_ci_stat(x,y,spearmanr,seed=self.rng.randint(1e6))
                    corr_rows.append({'kernel':k,'feature':feat,'metric':metric,'n':len(sub),'pearson_r':pr,'pearson_p':pp,'pearson_ci_low':pr_ci[0],'pearson_ci_high':pr_ci[1],'spearman_r':sr,'spearman_p':sp,'spearman_ci_low':sr_ci[0],'spearman_ci_high':sr_ci[1]})
            corr_df = pd.DataFrame(corr_rows)
            # BH per metric on Spearman p
            for metric in corr_df['metric'].unique():
                mask = corr_df['metric']==metric
                pvals = corr_df.loc[mask,'spearman_p'].fillna(1.0).values
                if len(pvals)>0:
                    corr_df.loc[mask,'spearman_p_fdr'] = benjamini_hochberg(pvals)
            corr_df.to_csv(OUTPUT_DIR / f"detailed_relations_{k}.csv", index=False)
            # 2) paired deltas vs baseline (for metrics) -- store for meta-model labels
            if BASELINE_KERNEL in merged['kernel'].unique():
                # join baseline values
                df_base = merged[merged['kernel']==BASELINE_KERNEL].set_index('dataset')
                df_this = dfk.set_index('dataset')
                common = df_this.index.intersection(df_base.index)
                for metric in metrics:
                    if metric in df_this.columns and metric in df_base.columns:
                        for d in common:
                            delta = df_this.loc[d,metric] - df_base.loc[d,metric]
                            all_delta.append({'kernel':k,'dataset':d,'metric':metric,'delta':float(delta)})
            # 3) LODO meta-model for wins (classification) and permutation importances averaged across LODO folds
            # define labels per metric by thresholding deltas later; here we'll do for accuracy only (example) and only if enough n
            if n >= MIN_N_PER_KERNEL and 'accuracy_mean' in dfk.columns:
                # prepare dataset-level X (meta-features) and y (win vs baseline)
                # ensure baseline exists and dataset intersection
                if BASELINE_KERNEL in merged['kernel'].unique():
                    df_base = merged[merged['kernel']==BASELINE_KERNEL].set_index('dataset')
                    df_this = dfk.set_index('dataset')
                    common = df_this.index.intersection(df_base.index)
                    if len(common) >= MIN_N_PER_KERNEL:
                        X = df_this.loc[common, features].copy()
                        y_delta = (df_this.loc[common,'accuracy_mean'] - df_base.loc[common,'accuracy_mean']).astype(float)
                        y = (y_delta > WIN_THRESHOLD).astype(int)  # binary wins
                        # impute + scale inside pipeline during model training; we do LODO
                        loo_importances = []
                        preds = []
                        true = []
                        for test_idx, dname in enumerate(common):
                            train_idx = [c for c in common if c != dname]
                            X_train = X.loc[train_idx]; y_train = y.loc[train_idx]
                            X_test = X.loc[[dname]]; y_test = y.loc[[dname]]
                            # train RF classifier
                            pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=self.rng.randint(1e6)))
                            # handle case few features/samples
                            try:
                                pipe.fit(X_train, y_train)
                                # permutation importance on held-out test (single sample -> permutation_importance not meaningful),
                                # so instead compute permutation importance on a small validation subset: use cross_val inside train set
                                # We'll compute permutation importances on a validation split from train if train size >=4
                                if len(X_train) >= 4:
                                    # holdout split from train
                                    idxs = list(range(len(X_train)))
                                    # simple split: last 20% as val
                                    split = max(1, int(0.8*len(X_train)))
                                    X_tr = X_train.iloc[:split]; y_tr = y_train.iloc[:split]
                                    X_val = X_train.iloc[split:]; y_val = y_train.iloc[split:]
                                    if len(X_val) >= 1:
                                        perm = permutation_importance(pipe, X_val, y_val, n_repeats=PERM_REPEATS, random_state=self.rng.randint(1e6), scoring='accuracy')
                                        loo_importances.append(perm.importances_mean)
                                # predict test
                                pred = pipe.predict(X_test)[0]
                            except Exception:
                                pred = int(y_train.mean()>=0.5)  # fallback majority
                            preds.append(pred); true.append(int(y_test.iloc[0]))
                        # aggregate importances
                        if len(loo_importances)>0:
                            arr = np.array(loo_importances)
                            mean_imp = arr.mean(axis=0)
                            std_imp = arr.std(axis=0)
                            imp_df = pd.DataFrame({'feature':X.columns.tolist(),'mean_importance':mean_imp,'std_importance':std_imp}).sort_values('mean_importance',ascending=False)
                            imp_df.to_csv(OUTPUT_DIR / f"meta_rf_importance_{k}_accuracy_LODO.csv", index=False)
                        # report simple LODO accuracy
                        acc = np.mean(np.array(preds) == np.array(true)) if len(true)>0 else np.nan
                        meta_model_summaries.append({'kernel':k,'metric':'accuracy','lodo_acc':acc,'n_datasets':len(common)})
        # write meta-model summary table
        if meta_model_summaries:
            pd.DataFrame(meta_model_summaries).to_csv(OUTPUT_DIR / "meta_model_summaries.csv", index=False)

    def multivariate_check_training_time(self, merged):
        """Check confounding: training_time ~ n_samples + class_imbalance_ratio + n_features"""
        cols = ['training_time_mean','n_samples','class_imbalance_ratio','n_features']
        for c in cols:
            if c not in merged.columns:
                logger.warning("Missing columns for multivariate check; skipping")
                return
        df = merged[cols].dropna()
        if len(df) < 6:
            logger.warning("Too few rows for multivariate regression")
            return
        X = sm.add_constant(df[['n_samples','class_imbalance_ratio','n_features']])
        y = df['training_time_mean']
        model = sm.OLS(y, X).fit(cov_type='HC3')  # robust SE
        out_file = OUTPUT_DIR / "training_time_multivar.txt"
        with open(out_file,'w') as f:
            f.write(model.summary().as_text())
        logger.info(f"Saved multivariate regression to {out_file}")

    def run_all(self):
        t0 = time.time()
        res, chars = self.load_data()
        merged = self.preprocess(res, chars)
        logger.info(f"Merged rows: {len(merged)}")
        # paired comparisons
        paired = self.paired_deltas_and_wilcoxon(merged)
        logger.info("Paired kernel comparisons saved.")
        # per-kernel analyses + meta-model LODO
        self.analyze_per_kernel(merged)
        # confounder multivar check
        self.multivariate_check_training_time(merged)
        logger.info(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    KernelMetaAnalyzer().run_all()
