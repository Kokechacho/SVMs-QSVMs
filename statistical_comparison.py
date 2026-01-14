import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
from Data.loader import load_all_datasets
from Kernels.all_kernels import build_single_kernel
import scipy.stats as stats
import time
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import warnings
import argparse
warnings.filterwarnings('ignore')

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BayesianOptimizationMethodology:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.load_base_config()
        self.load_datasets()

    # ----------------------------
    # Loading / helpers
    # ----------------------------
    def load_base_config(self):
        with open(self.config_path, 'r') as f:
            self.base_config = json.load(f)

    def load_datasets(self):
        """
        IMPORTANT: Do NOT scale the whole dataset here (avoids data leakage).
        We load raw X,y and scale later per-split / per-fold.
        """
        data_cfg = self.base_config['data']
        self.datasets = load_all_datasets(
            uci_list=data_cfg['uci'],
            libsvm_dir=Path(data_cfg['libsvm_dir']),
            libsvm_files=data_cfg['libsvm']
        )

        self.raw_datasets = []
        for dataset_name, X, y in self.datasets:
            # Save raw arrays; scaling will be applied later in a leakage-safe manner.
            self.raw_datasets.append((dataset_name, X, y))
            logger.info(f"Dataset {dataset_name} loaded (samples: {X.shape[0]}, features: {X.shape[1]})")
        logger.info(f"Loaded {len(self.raw_datasets)} raw datasets (no global scaling applied)")

    def get_kernel_function(self, kernel_name: str, params: Dict[str, Any]):
        """Return a callable kernel function built by build_single_kernel (or None on failure)."""
        try:
            kernel_params = params.copy()
            if 'gamma' in kernel_params:
                if isinstance(kernel_params['gamma'], str):
                    if kernel_params['gamma'] in ['auto', 'scale']:
                        pass
                    else:
                        try:
                            kernel_params['gamma'] = float(kernel_params['gamma'])
                        except (ValueError, TypeError):
                            kernel_params['gamma'] = 'scale'
                else:
                    kernel_params['gamma'] = float(kernel_params['gamma'])
            kernel_info = build_single_kernel(kernel_name, **kernel_params)
            return kernel_info['func'] if kernel_info else None
        except Exception as e:
            logger.error(f"Error building kernel {kernel_name}: {e}")
            return None

    def calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        if len(data) < 2:
            return (np.mean(data), np.mean(data))
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)
        if n >= 30:
            ci = stats.norm.interval(confidence, loc=mean, scale=sem)
        else:
            ci = stats.t.interval(confidence, n-1, loc=mean, scale=sem)
        return ci

    # ----------------------------
    # Cross-validation evaluation (used during search) -- SECURE SCALING PER FOLD
    # ----------------------------
    def evaluate_with_cross_validation(self, X: np.ndarray, y: np.ndarray,
                                       kernel_type: str, params: Dict[str, Any],
                                       cv_folds: int = 10, random_state: int = 42) -> Dict[str, float]:
        """
        Perform stratified K-fold CV to evaluate a kernel+params on (X,y).
        **Important:** for each CV fold we fit a scaler on X_train_fold and transform X_test_fold
        to avoid leakage from validation into training preprocessing.
        """
        try:
            kernel_func = self.get_kernel_function(kernel_type, params)
            if kernel_func is None:
                return {'accuracy': 0.0, 'f1_score': 0.0, 'support_vector_prop': 0.0, 'training_time': 0.0}

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            accuracy_scores = []
            f1_scores = []
            support_props = []
            training_times = []

            for train_idx, test_idx in cv.split(X, y):
                X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y[train_idx], y[test_idx]

                # --- SCALE safely: fit scaler on training fold only ---
                scaler = MinMaxScaler(feature_range=(-1, 1))
                X_train_fold_scaled = scaler.fit_transform(X_train_fold)
                X_test_fold_scaled = scaler.transform(X_test_fold)

                model = SVC(kernel=kernel_func, C=params.get('C', 1.0),
                            random_state=random_state, max_iter=10000)
                start_train = time.perf_counter()
                model.fit(X_train_fold_scaled, y_train_fold)
                training_times.append(time.perf_counter() - start_train)

                y_pred = model.predict(X_test_fold_scaled)
                accuracy_scores.append(accuracy_score(y_test_fold, y_pred))
                f1_scores.append(f1_score(y_test_fold, y_pred, average='weighted'))

                n_support_vectors = np.sum(model.n_support_)
                support_props.append(100 * n_support_vectors / len(X_train_fold))

            return {
                'accuracy': float(np.mean(accuracy_scores)),
                'accuracy_std': float(np.std(accuracy_scores)),
                'f1_score': float(np.mean(f1_scores)),
                'f1_std': float(np.std(f1_scores)),
                'support_vector_prop': float(np.mean(support_props)),
                'support_vector_std': float(np.std(support_props)),
                'training_time': float(np.mean(training_times)),
                'training_time_std': float(np.std(training_times))
            }
        except Exception as e:
            logger.warning(f"CV evaluation failed: {e}")
            return {'accuracy': 0.0, 'f1_score': 0.0, 'support_vector_prop': 0.0, 'training_time': 0.0}

    # ----------------------------
    # Hyperparameter optimization (works on the search set)
    # ----------------------------
    def optimize_hyperparams(self, X_search: np.ndarray, y_search: np.ndarray,
                             kernel_type: str, n_trials: int = 100,
                             cv_folds: int = 10, random_state: int = 42) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Run Optuna Bayesian optimization on the search set (X_search, y_search).
        Uses evaluate_with_cross_validation which already applies scaling per fold.
        """
        def objective(trial):
            params = {'C': trial.suggest_float('C', 0.001, 100)}
            if kernel_type == 'poly':
                params['degree'] = trial.suggest_int('degree', 1, 6)
                params['coef0'] = 1.0
                params['gamma'] = trial.suggest_float('gamma', 2**-6, 2**2)
            elif kernel_type == 'rbf':
                params['gamma'] = trial.suggest_float('gamma', 2**-6, 2**2)
            elif kernel_type == 'custom_hermite':
                params['degree'] = trial.suggest_int('degree', 1, 6)
            elif kernel_type == 'custom_gegen':
                params['degree'] = trial.suggest_int('degree', 1, 6)
                params['alpha'] = trial.suggest_float('alpha', -0.49, 1.5)
            elif kernel_type == 'custom_alsalam':
                params['degree'] = trial.suggest_int('degree', 1, 6)
                params['a'] = trial.suggest_categorical('a', [-1])
                params['q'] = trial.suggest_float('q', 0.01, 0.99, step=0.01)

            metrics = self.evaluate_with_cross_validation(
                X_search, y_search, kernel_type, params, cv_folds, random_state
            )
            return float(metrics['accuracy'])

        try:
            sampler = optuna.samplers.TPESampler(seed=random_state)
            study = optuna.create_study(direction='maximize',
                                       study_name=f"bayesian_opt_{kernel_type}_{random_state}",
                                       sampler=sampler)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            if study.best_trial:
                best_params = study.best_params
                best_metrics = self.evaluate_with_cross_validation(
                    X_search, y_search, kernel_type, best_params, cv_folds, random_state
                )
                return best_params, best_metrics
            else:
                return {}, {'accuracy': 0.0, 'f1_score': 0.0, 'support_vector_prop': 0.0, 'training_time': 0.0}
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            return {}, {'accuracy': 0.0, 'f1_score': 0.0, 'support_vector_prop': 0.0, 'training_time': 0.0}

    # ----------------------------
    # Final evaluation on held-out evaluation set (scale = fit on X_search)
    # ----------------------------
    def final_evaluate(self, X_search: np.ndarray, y_search: np.ndarray,
                       X_eval: np.ndarray, y_eval: np.ndarray,
                       kernel_type: str, best_params: Dict[str, Any],
                       full_dataset_size: int, random_state: int = 42) -> Tuple[float, float, float, float]:
        """
        Train on the entire search set with best_params and evaluate on X_eval.
        Critically: fit the scaler on X_search (NOT on X_eval), then transform X_eval with it.
        """
        try:
            # Fit scaler ONCE on the whole search set and apply to eval (no leakage)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_search_scaled = scaler.fit_transform(X_search)
            X_eval_scaled = scaler.transform(X_eval)

            kernel_func_final = self.get_kernel_function(kernel_type, best_params)
            model_final = SVC(kernel=kernel_func_final,
                              C=best_params.get('C', 1.0),
                              random_state=random_state,
                              max_iter=10000)
            start_train_final = time.perf_counter()
            model_final.fit(X_search_scaled, y_search)
            training_time_final = time.perf_counter() - start_train_final

            y_pred_eval = model_final.predict(X_eval_scaled)
            acc_eval = accuracy_score(y_eval, y_pred_eval)
            f1_eval = f1_score(y_eval, y_pred_eval, average='weighted')

            n_support_vectors_final = np.sum(model_final.n_support_)
            support_prop_eval = 100 * n_support_vectors_final / full_dataset_size

            return float(acc_eval), float(f1_eval), float(support_prop_eval), float(training_time_final)
        except Exception as e:
            logger.warning(f"Final evaluation failed: {e}")
            return 0.0, 0.0, 0.0, 0.0

    # ----------------------------
    # Single trial runner (partition -> optimize -> final eval)
    # ----------------------------
    def run_trial(self, dataset_name: str, X: np.ndarray, y: np.ndarray,
                  kernel_type: str, trial_idx: int, random_seed: int,
                  n_optimization_trials: int, cv_folds: int, test_size: float):
        """
        Performs:
         1) stratified split (search / eval) -> NO scaling yet
         2) hyperparameter optimization on search set (CV with per-fold scaling)
         3) final evaluation on eval set (scaling fitted on full search set)
        """
        # 1) partition (no global scaling before this!)
        X_search, X_eval, y_search, y_eval = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_seed
        )

        start_time = time.perf_counter()
        # 2) hyperparameter search on search set
        best_params, best_metrics_search = self.optimize_hyperparams(
            X_search, y_search, kernel_type,
            n_trials=n_optimization_trials, cv_folds=cv_folds, random_state=random_seed
        )
        optimization_time = time.perf_counter() - start_time

        if not best_params:
            logger.warning(f"   Optimization failed for trial {trial_idx} (dataset={dataset_name}, kernel={kernel_type})")
            return None, None

        # 3) final evaluation on held-out eval set (scaler fit on X_search)
        acc_eval, f1_eval, support_prop_eval, training_time_final = self.final_evaluate(
            X_search, y_search, X_eval, y_eval, kernel_type, best_params,
            full_dataset_size=len(X_search), random_state=random_seed
        )

        # Build history and detailed result entries
        optimization_history_entry = {
            'dataset': dataset_name,
            'kernel': kernel_type,
            'trial': trial_idx,
            'optimization_time': optimization_time,
            'training_time_search_cv': best_metrics_search.get('training_time', 0.0),
            'best_search_accuracy': best_metrics_search.get('accuracy', 0.0),
            'best_search_f1_score': best_metrics_search.get('f1_score', 0.0),
            'best_search_support_vector_prop': best_metrics_search.get('support_vector_prop', 0.0),
            'final_eval_accuracy': acc_eval,
            'final_eval_f1_score': f1_eval,
            'final_eval_support_vector_prop': support_prop_eval,
            'final_training_time': training_time_final,
            'best_params': best_params,
            'random_seed': random_seed,
            'n_optimization_trials': n_optimization_trials,
            'cv_folds': cv_folds
        }

        detailed_result_entry = {
            'dataset': dataset_name,
            'kernel': kernel_type,
            'trial': trial_idx,
            'accuracy': acc_eval,
            'search_accuracy': best_metrics_search.get('accuracy', 0.0),
            'accuracy_std': best_metrics_search.get('accuracy_std', 0.0),
            'f1_score': f1_eval,
            'search_f1_score': best_metrics_search.get('f1_score', 0.0),
            'f1_std': best_metrics_search.get('f1_std', 0.0),
            'support_vector_prop': support_prop_eval,
            'search_support_vector_prop': best_metrics_search.get('support_vector_prop', 0.0),
            'training_time': training_time_final,
            'training_time_std': best_metrics_search.get('training_time_std', 0.0),
            'optimization_time': optimization_time,
            'random_seed': random_seed,
            'best_params': str(best_params),
            'n_optimization_trials': n_optimization_trials,
            'cv_folds': cv_folds,
            'optimization_method': 'bayesian'
        }

        logger.info(f"   Trial {trial_idx+1} completed - Eval Acc: {acc_eval:.4f} - Eval F1: {f1_eval:.4f} - Support%: {support_prop_eval:.4f}")
        return optimization_history_entry, detailed_result_entry

    # ----------------------------
    # Kernel on dataset runner (loops trials)
    # ----------------------------
    def run_kernel_on_dataset(self, dataset_name: str, X: np.ndarray, y: np.ndarray,
                              kernel_type: str, n_experimental_trials: int,
                              n_optimization_trials: int, cv_folds: int, test_size: float):
        optimization_history = []
        detailed_results = []
        trial_accuracies = []
        trial_f1_scores = []
        trial_support_vector_props = []
        trial_training_times = []
        trial_optimization_times = []

        for trial_idx in range(n_experimental_trials):
            random_seed = 42 + trial_idx
            logger.info(f"   Trial {trial_idx+1}/{n_experimental_trials} - Dataset: {dataset_name}, Kernel: {kernel_type}")
            hist_entry, det_entry = self.run_trial(
                dataset_name, X, y, kernel_type, trial_idx, random_seed,
                n_optimization_trials, cv_folds, test_size
            )
            if hist_entry is None:
                continue
            optimization_history.append(hist_entry)
            detailed_results.append(det_entry)
            trial_accuracies.append(det_entry['accuracy'])
            trial_f1_scores.append(det_entry['f1_score'])
            trial_support_vector_props.append(det_entry['support_vector_prop'])
            trial_training_times.append(det_entry['training_time'])
            trial_optimization_times.append(det_entry['optimization_time'])

        # summary stats for this dataset-kernel (based on final eval across trials)
        summary = None
        if trial_accuracies:
            acc_mean = np.mean(trial_accuracies)
            acc_std = np.std(trial_accuracies)
            acc_median = np.median(trial_accuracies)
            acc_ci = self.calculate_confidence_interval(trial_accuracies, 0.95)

            f1_mean = np.mean(trial_f1_scores)
            f1_std = np.std(trial_f1_scores)
            f1_median = np.median(trial_f1_scores)

            support_mean = np.mean(trial_support_vector_props)
            support_std = np.std(trial_support_vector_props)
            support_median = np.median(trial_support_vector_props)

            train_mean = np.mean(trial_training_times)
            train_std = np.std(trial_training_times)
            train_median = np.median(trial_training_times)

            opt_mean = np.mean(trial_optimization_times) if trial_optimization_times else 0.0
            opt_std = np.std(trial_optimization_times) if trial_optimization_times else 0.0
            opt_median = np.median(trial_optimization_times) if trial_optimization_times else 0.0

            summary = {
                'dataset': dataset_name,
                'kernel': kernel_type,
                'n_trials': len(trial_accuracies),
                'accuracy_mean': acc_mean,
                'accuracy_std': acc_std,
                'accuracy_median': acc_median,
                'accuracy_ci_lower': acc_ci[0],
                'accuracy_ci_upper': acc_ci[1],
                'accuracy_min': np.min(trial_accuracies),
                'accuracy_max': np.max(trial_accuracies),
                'f1_mean': f1_mean,
                'f1_std': f1_std,
                'f1_median': f1_median,
                'support_vector_prop_mean': support_mean,
                'support_vector_prop_std': support_std,
                'support_vector_prop_median': support_median,
                'training_time_mean': train_mean,
                'training_time_std': train_std,
                'training_time_median': train_median,
                'optimization_time_mean': opt_mean,
                'optimization_time_std': opt_std,
                'optimization_time_median': opt_median,
                'n_optimization_trials': n_optimization_trials,
                'cv_folds': cv_folds,
                'optimization_method': 'bayesian'
            }

            logger.info(f"Final Results {dataset_name}-{kernel_type}: Accuracy = {acc_mean:.4f} ± {acc_std:.4f} (avg of {len(trial_accuracies)} trials)")
        else:
            logger.warning(f"No successful trials completed for {dataset_name}-{kernel_type}")

        return summary, detailed_results, optimization_history

    # ----------------------------
    # Main execution (datasets x kernels)
    # ----------------------------
    def execute_bayesian_optimization_methodology(self,
                                                  n_experimental_trials: int = 35,
                                                  n_optimization_trials: int = 100,
                                                  cv_folds: int = 10,
                                                  test_size: float = 0.30):
        logger.info("BAYESIAN OPTIMIZATION METHODOLOGY")
        logger.info(f"  - {n_experimental_trials} experimental trials")
        logger.info(f"  - Partition: {1-test_size:.2f} search / {test_size:.2f} eval (stratified)")
        logger.info(f"  - {n_optimization_trials} optimization evaluations per kernel (Bayesian)")
        logger.info(f"  - {cv_folds}-fold CV (performed on search set during optimization with per-fold scaling)")

        kernels_to_evaluate = ['linear', 'poly', 'rbf', 'custom_hermite', 'custom_gegen', 'custom_alsalam']

        all_results = []
        all_detailed = []
        all_optimization_history = []

        for dataset_name, X, y in self.raw_datasets:
            logger.info(f"Dataset: {dataset_name} (samples: {X.shape[0]}, features: {X.shape[1]})")
            for kernel_type in kernels_to_evaluate:
                logger.info(f"Kernel: {kernel_type}")
                summary, detailed_results, optimization_history = self.run_kernel_on_dataset(
                    dataset_name, X, y, kernel_type,
                    n_experimental_trials=n_experimental_trials,
                    n_optimization_trials=n_optimization_trials,
                    cv_folds=cv_folds,
                    test_size=test_size
                )
                if summary:
                    all_results.append(summary)
                if detailed_results:
                    all_detailed.extend(detailed_results)
                if optimization_history:
                    all_optimization_history.extend(optimization_history)

        self.save_results(all_results, all_detailed, all_optimization_history)
        return all_results, all_detailed, all_optimization_history

    # ----------------------------
    # Saving and reporting 
    # ----------------------------
    def save_results(self, summary_results: List[Dict], detailed_results: List[Dict],
                     optimization_history: List[Dict]):
        output_dir = Path("bayesian_optimization_methodology")
        output_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if summary_results:
            df_summary = pd.DataFrame(summary_results)
            summary_file = output_dir / f"bayesian_methodology_summary_{timestamp}.csv"
            df_summary.to_csv(summary_file, index=False)
            logger.info(f"Summary saved to {summary_file}")

        if detailed_results:
            df_detailed = pd.DataFrame(detailed_results)
            detailed_file = output_dir / f"bayesian_methodology_detailed_{timestamp}.csv"
            df_detailed.to_csv(detailed_file, index=False)
            logger.info(f"Detailed results saved to {detailed_file}")

        if optimization_history:
            df_optimization = pd.DataFrame(optimization_history)
            optimization_file = output_dir / f"bayesian_methodology_optimization_{timestamp}.csv"
            df_optimization.to_csv(optimization_file, index=False)
            logger.info(f"Optimization history saved to {optimization_file}")

        self.generate_report(summary_results, output_dir, timestamp)

    def generate_report(self, summary_results: List[Dict], output_dir: Path, timestamp: str):
        if not summary_results:
            return
        df = pd.DataFrame(summary_results)
        report_lines = []
        report_lines.append("=" * 120)
        report_lines.append("REPORT - BAYESIAN OPTIMIZATION METHODOLOGY")
        report_lines.append("=" * 120)
        report_lines.append("METHODOLOGY:")
        report_lines.append(f"  - {len(df)} combinations (dataset-kernel) summarized")
        report_lines.append("  - Each experiment: stratified split search/eval, optimization on search (CV with per-fold scaling), eval on held-out.")
        report_lines.append(f"Timestamp: {timestamp}")
        report_lines.append("")
        for dataset in df['dataset'].unique():
            report_lines.append(f"DATASET: {dataset}")
            report_lines.append("-" * 100)
            df_ds = df[df['dataset'] == dataset].sort_values('accuracy_mean', ascending=False)
            for _, row in df_ds.iterrows():
                report_lines.append(
                    f"{row['kernel']:15} | "
                    f"Accuracy: {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
                    f"F1-Score: {row['f1_mean']:.4f} | "
                    f"Support Vectors: {row['support_vector_prop_mean']:.4f} | "
                    f"Training Time: {row['training_time_mean']:.4f}s"
                )
        report_lines.append("\n" + "=" * 120)
        report_lines.append("BEST OVERALL RESULT")
        report_lines.append("=" * 120)
        best_overall = df.loc[df['accuracy_mean'].idxmax()]
        report_lines.append(f"Combination: {best_overall['dataset']} - {best_overall['kernel']}")
        report_lines.append(f"Accuracy: {best_overall['accuracy_mean']:.4f} ± {best_overall['accuracy_std']:.4f}")
        report_lines.append(f"F1-Score: {best_overall['f1_mean']:.4f}")
        report_lines.append(f"Support Vectors: {best_overall['support_vector_prop_mean']:.4f}")
        report_lines.append(f"Training Time: {best_overall['training_time_mean']:.4f}s")
        report_lines.append(f"Optimization Time: {best_overall['optimization_time_mean']:.1f}s")
        report_lines.append(f"Number of trials: {best_overall['n_trials']}")
        report_file = output_dir / f"bayesian_methodology_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        logger.info(f"Report saved to {report_file}")
        print('\n'.join(report_lines[-20:]))

# ----------------------------
# CLI and main
# ----------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description='Bayesian Optimization Methodology')
    parser.add_argument('--n_trials', type=int, default=35, help='Number of experimental trials (default: 35)')
    parser.add_argument('--n_optimization_trials', type=int, default=100, help='Optimization evaluations (default: 100)')
    parser.add_argument('--cv_folds', type=int, default=10, help='Cross-validation folds (default: 10)')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    parser.add_argument('--test_size', type=float, default=0.30, help='Proportion for evaluation set (default: 0.30)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    evaluator = BayesianOptimizationMethodology(config_path=args.config)
    logger.info("INITIATING BAYESIAN OPTIMIZATION METHODOLOGY")
    try:
        start_time = time.time()
        summary_results, detailed_results, optimization_history = evaluator.execute_bayesian_optimization_methodology(
            n_experimental_trials=args.n_trials,
            n_optimization_trials=args.n_optimization_trials,
            cv_folds=args.cv_folds,
            test_size=args.test_size
        )
        total_time = time.time() - start_time
        if summary_results:
            logger.info(f"Evaluation completed in {total_time:.1f}s ({total_time/60:.1f}min)!")
            logger.info("Results saved to 'bayesian_optimization_methodology/'")
            best_result = max(summary_results, key=lambda x: x['accuracy_mean'])
            logger.info(f"Best result: {best_result['dataset']}-{best_result['kernel']} "
                       f"Accuracy: {best_result['accuracy_mean']:.4f} "
                       f"F1-Score: {best_result['f1_mean']:.4f} "
                       f"Training Time: {best_result['training_time_mean']:.4f}s")
        else:
            logger.error("Evaluation produced no results")
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
