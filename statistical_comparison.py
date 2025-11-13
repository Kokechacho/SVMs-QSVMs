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
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import warnings
import argparse
warnings.filterwarnings('ignore')

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Main class: Evaluation pipeline with Bayesian Optimization
class BayesianOptimizationMethodology:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.load_base_config()
        self.load_datasets()
        
    def load_base_config(self):
        """Load base configuration"""
        with open(self.config_path, 'r') as f:
            self.base_config = json.load(f)
    
    def load_datasets(self):
        """Load all datasets and scale to [-1, 1]"""
        data_cfg = self.base_config['data']
        self.datasets = load_all_datasets(
            uci_list=data_cfg['uci'],
            libsvm_dir=Path(data_cfg['libsvm_dir']),
            libsvm_files=data_cfg['libsvm']
        )
        
        # Scale all datasets to [-1, 1]
        self.scaled_datasets = []
        self.scalers = {}
        
        for dataset_name, X, y in self.datasets:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_scaled = scaler.fit_transform(X)
            
            self.scaled_datasets.append((dataset_name, X_scaled, y))
            self.scalers[dataset_name] = scaler
            
            min_val = np.min(X_scaled)
            max_val = np.max(X_scaled)
            logger.info(f"Dataset {dataset_name} scaled to [{min_val:.3f}, {max_val:.3f}]")
        
        logger.info(f"Loaded and scaled {len(self.scaled_datasets)} datasets to [-1, 1]")
    
    # Retrieve kernel function using build_single_kernel
    def get_kernel_function(self, kernel_name: str, params: Dict[str, Any]):
        """Get kernel function with specific parameters"""
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
        """Calculate confidence interval"""
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
    
    # Given a kernel perform stratified K-fold CV (same evaluation for the trial) training an SVM and return CV-aggregated metrics
    def evaluate_with_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                    kernel_type: str, params: Dict[str, Any],
                                    cv_folds: int = 10, random_state: int = 42) -> Dict[str, float]:
        """
        Evaluate hyperparameter configuration using 10-fold cross-validation
        """
        try:
            kernel_func = self.get_kernel_function(kernel_type, params)
            if kernel_func is None:
                return {'accuracy': 0.0, 'f1_score': 0.0, 'support_vector_prop': 0.0, 'training_time': 0.0}
            
            # Use the SAME fold partition for all evaluations in this trial
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            accuracy_scores = []
            f1_scores = []
            support_props = []
            training_times = []

            N = len(X)  # total number of samples in the whole dataset (denominator)
            
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model = SVC(
                    kernel=kernel_func,
                    C=params.get('C', 1.0),
                    random_state=random_state,
                    max_iter=10000
                )
                
                # Train and measure training time
                start_train = time.perf_counter()
                model.fit(X_train, y_train)
                training_time = time.perf_counter() - start_train
                training_times.append(training_time)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics for this fold
                accuracy_scores.append(accuracy_score(y_test, y_pred))
                f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
                
                # Calculate support vectors proportion
                n_support_vectors = np.sum(model.n_support_)
                support_prop = 100 * n_support_vectors / N
                support_props.append(support_prop)

            return {
                'accuracy': np.mean(accuracy_scores),
                'accuracy_std': np.std(accuracy_scores),
                'f1_score': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'support_vector_prop': np.mean(support_props),
                'support_vector_std': np.std(support_props),
                'training_time': np.mean(training_times),
                'training_time_std': np.std(training_times)
            }
        except Exception as e:
            logger.warning(f"CV evaluation failed: {e}")
            return {'accuracy': 0.0, 'f1_score': 0.0, 'support_vector_prop': 0.0, 'training_time': 0.0}
    
    # Run Bayesian optimization process for a given dataset and kernel using fixed CV folds for the trial
    def bayesian_optimization_process(self, X: np.ndarray, y: np.ndarray, 
                                kernel_type: str, n_trials: int = 100,
                                cv_folds: int = 10, random_state: int = 42) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Bayesian optimization process with Optuna (TPESampler)
        - Same 10-fold partition for ALL evaluations in this trial
        - n_trials evaluations (default 100)
        - Selects best configuration based on 10-fold CV
        """
        
        def objective(trial):
            # Parameter ranges identical to original study
            params = {'C': trial.suggest_float('C', 0.001, 100)}
            
            if kernel_type == 'poly':
                params['degree'] = trial.suggest_int('degree', 1, 6)
                params['coef0'] = 1.0  # Fixed as in original study
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
            
            # Evaluate with 10-fold CV on the SAME partition of this trial!!!
            metrics = self.evaluate_with_cross_validation(
                X, y, kernel_type, params, cv_folds, random_state
            )
            return metrics['accuracy']

        try:
            # Use TPESampler for Bayesian Optimization
            sampler = optuna.samplers.TPESampler(seed=random_state)
            
            study = optuna.create_study(
                direction='maximize',
                study_name=f"bayesian_opt_{kernel_type}_{random_state}",
                sampler=sampler
            )
            
            # OPTIMIZATION
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            # EVALUATION
            if study.best_trial:
                best_params = study.best_params
                # Final evaluation with same folds (to report metrics)
                best_metrics = self.evaluate_with_cross_validation(
                    X, y, kernel_type, best_params, cv_folds, random_state
                )
                return best_params, best_metrics
            else:
                return {}, {'accuracy': 0.0, 'f1_score': 0.0, 'support_vector_prop': 0.0, 'training_time': 0.0}
                
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            return {}, {'accuracy': 0.0, 'f1_score': 0.0, 'support_vector_prop': 0.0, 'training_time': 0.0}
    
    def execute_bayesian_optimization_methodology(self, 
                               n_experimental_trials: int = 35,
                               n_optimization_trials: int = 100,
                               cv_folds: int = 10):
        """
        Executes the methodology with Bayesian Optimization.
        
        1. 35 experimental trials with different seeds
        2. Each trial: DIFFERENT 10-fold partition and DIFFERENT seed for optimization
        3. For each kernel: Optimize evaluating n_optimization_trials configurations with 10-fold CV
        4. Select best configuration per kernel
        5. Report average of 35 best kernels
        """
        
        logger.info("BAYESIAN OPTIMIZATION METHODOLOGY")
        logger.info("Methodology characteristics:")
        logger.info(f"  - {n_experimental_trials} experimental trials with different seeds")
        logger.info(f"  - Each trial: different 10-fold partition")
        logger.info(f"  - Each evaluation: 10-fold cross-validation")
        logger.info(f"  - {n_optimization_trials} optimization evaluations per kernel (Bayesian Optimization)")
        logger.info("  - Same folds for all kernels in the same trial")
        logger.info("  - Average of 35 best kernels")
        
        all_results = []
        detailed_results = []
        optimization_history = []
        
        kernels_to_evaluate = ['linear', 'poly', 'rbf', 'custom_hermite', 'custom_gegen', 'custom_alsalam']
        
        for dataset_name, X, y in self.scaled_datasets:
            logger.info(f"Dataset: {dataset_name} (samples: {X.shape[0]}, features: {X.shape[1]})")
            
            for kernel_type in kernels_to_evaluate:
                logger.info(f"Kernel: {kernel_type}")
                
                trial_accuracies = []
                trial_f1_scores = []
                trial_support_vector_props = []
                trial_training_times = []
                trial_optimization_times = []
                
                for trial_idx in range(n_experimental_trials):
                    try:
                        # DIFFERENT seed for each trial
                        random_seed = 42 + trial_idx
                        
                        logger.info(f"   Trial {trial_idx+1}/{n_experimental_trials} - Dataset: {dataset_name}, Kernel: {kernel_type}")
                        
                        # BAYESIAN OPTIMIZATION with specific 10-fold partition for this trial
                        start_time = time.perf_counter()
                        
                        best_params, best_metrics = self.bayesian_optimization_process(
                            X, y, kernel_type,
                            n_trials=n_optimization_trials,
                            cv_folds=cv_folds,
                            random_state=random_seed
                        )
                        
                        optimization_time = time.perf_counter() - start_time
                        
                        # Save optimization history
                        optimization_history.append({
                            'dataset': dataset_name,
                            'kernel': kernel_type,
                            'trial': trial_idx,
                            'optimization_time': optimization_time,
                            'training_time': best_metrics['training_time'],
                            'best_accuracy': best_metrics['accuracy'],
                            'best_f1_score': best_metrics['f1_score'],
                            'support_vector_prop': best_metrics['support_vector_prop'],
                            'best_params': best_params,
                            'random_seed': random_seed,
                            'n_optimization_trials': n_optimization_trials,
                            'cv_folds': cv_folds
                        })
                        
                        if not best_params:
                            logger.warning(f"   Optimization failed for trial {trial_idx}")
                            continue
                        
                        # Store trial results
                        trial_accuracies.append(best_metrics['accuracy'])
                        trial_f1_scores.append(best_metrics['f1_score'])
                        trial_support_vector_props.append(best_metrics['support_vector_prop'])
                        trial_training_times.append(best_metrics['training_time'])
                        trial_optimization_times.append(optimization_time)
                        
                        # Save detailed result
                        detailed_results.append({
                            'dataset': dataset_name,
                            'kernel': kernel_type,
                            'trial': trial_idx,
                            'accuracy': best_metrics['accuracy'],
                            'accuracy_std': best_metrics.get('accuracy_std', 0.0),
                            'f1_score': best_metrics['f1_score'],
                            'f1_std': best_metrics.get('f1_std', 0.0),
                            'support_vector_prop': best_metrics['support_vector_prop'],
                            'support_vector_std': best_metrics.get('support_vector_std', 0.0),
                            'training_time': best_metrics['training_time'],
                            'training_time_std': best_metrics.get('training_time_std', 0.0),
                            'optimization_time': optimization_time,
                            'random_seed': random_seed,
                            'best_params': str(best_params),
                            'n_optimization_trials': n_optimization_trials,
                            'cv_folds': cv_folds,
                            'optimization_method': 'bayesian'
                        })
                        
                        logger.info(f"   Trial {trial_idx+1} completed - "
                                   f"Accuracy: {best_metrics['accuracy']:.4f} - "
                                   f"F1-Score: {best_metrics['f1_score']:.4f} - "
                                   f"Support Vectors: {best_metrics['support_vector_prop']:.4f} - "
                                   f"Training Time: {best_metrics['training_time']:.4f}s - "
                                   f"Optimization Time: {optimization_time:.1f}s")
                        
                    except Exception as e:
                        logger.warning(f"Trial {trial_idx} failed: {e}")
                        continue
                
                # Calculate final statistics for this dataset-kernel combination
                if trial_accuracies:
                    acc_mean = np.mean(trial_accuracies)
                    acc_std = np.std(trial_accuracies)
                    acc_median = np.median(trial_accuracies)
                    acc_ci = self.calculate_confidence_interval(trial_accuracies, 0.95)
                    
                    f1_mean = np.mean(trial_f1_scores)
                    f1_std = np.std(trial_f1_scores)
                    f1_median = np.median(trial_f1_scores)
                    
                    support_vector_prop_mean = np.mean(trial_support_vector_props)
                    support_vector_prop_std = np.std(trial_support_vector_props)
                    support_vector_prop_median = np.median(trial_support_vector_props)
                    
                    training_time_mean = np.mean(trial_training_times)
                    training_time_std = np.std(trial_training_times)
                    training_time_median = np.median(trial_training_times)
                    
                    optimization_time_mean = np.mean(trial_optimization_times)
                    optimization_time_std = np.std(trial_optimization_times)
                    optimization_time_median = np.median(trial_optimization_times)
                    
                    result = {
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
                        'support_vector_prop_mean': support_vector_prop_mean,
                        'support_vector_prop_std': support_vector_prop_std,
                        'support_vector_prop_median': support_vector_prop_median,
                        'training_time_mean': training_time_mean,
                        'training_time_std': training_time_std,
                        'training_time_median': training_time_median,
                        'optimization_time_mean': optimization_time_mean,
                        'optimization_time_std': optimization_time_std,
                        'optimization_time_median': optimization_time_median,
                        'n_optimization_trials': n_optimization_trials,
                        'cv_folds': cv_folds,
                        'optimization_method': 'bayesian'
                    }
                    
                    all_results.append(result)
                    
                    logger.info(f"Final Results {dataset_name}-{kernel_type}: "
                               f"Accuracy = {acc_mean:.4f} ± {acc_std:.4f} | "
                               f"F1-Score = {f1_mean:.4f} | "
                               f"Support Vectors = {support_vector_prop_mean:.4f} | "
                               f"Training Time = {training_time_mean:.4f}s | "
                               f"Optimization Time = {optimization_time_mean:.1f}s "
                               f"(average of {len(trial_accuracies)} trials)")
                else:
                    logger.warning(f"No successful trials completed for {dataset_name}-{kernel_type}")
        
        # Save results
        self.save_results(all_results, detailed_results, optimization_history)
        return all_results, detailed_results, optimization_history
    
    def save_results(self, summary_results: List[Dict], detailed_results: List[Dict], 
                    optimization_history: List[Dict]):
        """Save all results"""
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
        """Generate detailed report"""
        if not summary_results:
            return
            
        df = pd.DataFrame(summary_results)
        
        report_lines = []
        report_lines.append("=" * 120)
        report_lines.append("REPORT - BAYESIAN OPTIMIZATION METHODOLOGY")
        report_lines.append("=" * 120)
        report_lines.append("METHODOLOGY:")
        report_lines.append("  - 35 experimental trials with different seeds")
        report_lines.append("  - Each trial: different 10-fold partition")
        report_lines.append("  - Optimization: Bayesian Optimization (Optuna) with 100 evaluations")
        report_lines.append("  - Each evaluation: 10-fold cross-validation")
        report_lines.append("  - Same folds for all kernels in a trial")
        report_lines.append("  - Average of the 35 best kernels")
        report_lines.append(f"Total combinations evaluated: {len(df)}")
        report_lines.append(f"Timestamp: {timestamp}")
        report_lines.append("")
        
        # Results by dataset
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
        
        # Best global result
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
        
        # Save report
        report_file = output_dir / f"bayesian_methodology_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report saved to {report_file}")
        print('\n'.join(report_lines[-20:]))

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Bayesian Optimization Methodology')
    
    parser.add_argument('--n_trials', type=int, default=35, 
                       help='Number of experimental trials (default: 35)')
    parser.add_argument('--n_optimization_trials', type=int, default=100,
                       help='Optimization evaluations (default: 100 for Bayesian Optimization)')
    parser.add_argument('--cv_folds', type=int, default=10,
                       help='Cross-validation folds (default: 10)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    evaluator = BayesianOptimizationMethodology(config_path=args.config)
    
    logger.info("INITIATING BAYESIAN OPTIMIZATION METHODOLOGY")
    logger.info("Methodology characteristics:")
    logger.info(f"  - {args.n_trials} experimental trials with different seeds")
    logger.info(f"  - Each trial: different 10-fold partition")
    logger.info(f"  - {args.n_optimization_trials} optimization evaluations (Bayesian Optimization)")
    logger.info(f"  - {args.cv_folds}-fold CV for each evaluation")
    logger.info("  - Same folds for all kernels in a trial")
    logger.info("  - Average of 35 best kernels")
    
    try:
        start_time = time.time()
        summary_results, detailed_results, optimization_history = evaluator.execute_bayesian_optimization_methodology(
            n_experimental_trials=args.n_trials,
            n_optimization_trials=args.n_optimization_trials,
            cv_folds=args.cv_folds
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