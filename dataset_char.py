import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import argparse
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetCharacterizer:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.datasets = []
        
    def load_datasets(self):
        """Load all datasets using the same loader as in optimization script"""
        try:
            from Data.loader import load_all_datasets
            import json
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            data_cfg = config['data']
            self.datasets = load_all_datasets(
                uci_list=data_cfg['uci'],
                libsvm_dir=Path(data_cfg['libsvm_dir']),
                libsvm_files=data_cfg['libsvm']
            )
            
            logger.info(f"Successfully loaded {len(self.datasets)} datasets")
            return self.datasets
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise
    
    def safe_convert_labels(self, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Safely convert labels to integers and return mapping"""
        original_dtype = y.dtype
        
        if y.dtype.kind in ['i', 'u']:  # Already integer types
            y_int = y.astype(int)
            unique_labels = np.unique(y_int)
            label_mapping = {label: label for label in unique_labels}
            return y_int, label_mapping
        else:
            # Use LabelEncoder for non-integer labels
            le = LabelEncoder()
            y_int = le.fit_transform(y)
            label_mapping = {i: original_label for i, original_label in enumerate(le.classes_)}
            return y_int, label_mapping
    
    def compute_dataset_characteristics(self, X: np.ndarray, y: np.ndarray, dataset_name: str) -> Dict:
        """Compute comprehensive dataset characteristics with proper label handling"""
        n_samples, n_features = X.shape
        
        # Safely convert labels to integers and get mapping
        y_int, label_mapping = self.safe_convert_labels(y)
        
        # Get unique classes and their counts
        unique_classes, class_counts = np.unique(y_int, return_counts=True)
        n_classes = len(unique_classes)
        class_percentages = (class_counts / n_samples) * 100
        
        # Feature statistics
        X_float = X.astype(float)
        feature_means = np.mean(X_float, axis=0)
        feature_stds = np.std(X_float, axis=0)
        feature_vars = np.var(X_float, axis=0)
        
        # Feature ranges and variability
        feature_mins = np.min(X_float, axis=0)
        feature_maxs = np.max(X_float, axis=0)
        feature_ranges = feature_maxs - feature_mins
        
        # Overall dataset statistics
        overall_mean = np.mean(X_float)
        overall_std = np.std(X_float)
        overall_var = np.var(X_float)
        
        # Feature with maximum mean (most dominant feature)
        max_mean_feature_idx = np.argmax(feature_means)
        max_mean_value = feature_means[max_mean_feature_idx]
        
        # Feature with maximum variability (highest std)
        max_std_feature_idx = np.argmax(feature_stds)
        max_std_value = feature_stds[max_std_feature_idx]
        
        # Feature with minimum variability (most constant)
        valid_stds = feature_stds[feature_stds > 1e-8]  # Exclude near-constant features
        if len(valid_stds) > 0:
            min_std_feature_idx = np.argmin(feature_stds)
            min_std_value = feature_stds[min_std_feature_idx]
        else:
            min_std_feature_idx = -1
            min_std_value = 0.0
        
        # Correlation structure
        try:
            correlation_matrix = np.corrcoef(X_float.T)
            np.fill_diagonal(correlation_matrix, 0)  # Remove self-correlations
            max_feature_correlation = np.max(np.abs(correlation_matrix))
            mean_feature_correlation = np.mean(np.abs(correlation_matrix))
        except:
            max_feature_correlation = 0.0
            mean_feature_correlation = 0.0
        
        # Build class distribution description with original labels
        class_distribution_desc = []
        for class_idx, (original_label, count, percentage) in enumerate(zip(unique_classes, class_counts, class_percentages)):
            mapped_label = label_mapping.get(original_label, original_label)
            class_distribution_desc.append(f"{mapped_label}: {count} samples ({percentage:.1f}%)")
        
        characteristics = {
            'dataset_name': dataset_name,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'samples_per_feature_ratio': n_samples / n_features if n_features > 0 else float('inf'),
            
            # Overall statistics
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'overall_variance': overall_var,
            'data_range_min': np.min(X_float),
            'data_range_max': np.max(X_float),
            'data_range_total': np.max(X_float) - np.min(X_float),
            
            # Feature statistics
            'mean_feature_mean': np.mean(feature_means),
            'std_feature_mean': np.std(feature_means),
            'mean_feature_std': np.mean(feature_stds),
            'std_feature_std': np.std(feature_stds),
            'mean_feature_variance': np.mean(feature_vars),
            'mean_feature_range': np.mean(feature_ranges),
            
            # Extreme features
            'max_mean_feature_value': max_mean_value,
            'max_mean_feature_index': int(max_mean_feature_idx),
            'max_std_feature_value': max_std_value,
            'max_std_feature_index': int(max_std_feature_idx),
            'min_std_feature_value': min_std_value,
            'min_std_feature_index': int(min_std_feature_idx),
            
            # Correlation
            'max_feature_correlation': max_feature_correlation,
            'mean_feature_correlation': mean_feature_correlation,
            
            # Constant features
            'n_constant_features': int(np.sum(feature_stds < 1e-8)),
            'proportion_constant_features': np.mean(feature_stds < 1e-8),
            
            # Class information
            'majority_class_size': np.max(class_counts),
            'minority_class_size': np.min(class_counts),
            'majority_class_percentage': np.max(class_percentages),
            'minority_class_percentage': np.min(class_percentages),
            'class_imbalance_ratio': np.min(class_counts) / np.max(class_counts) if np.max(class_counts) > 0 else 0,
            'class_entropy': entropy(class_counts / n_samples) if n_classes > 1 else 0,
            'class_distribution_description': '; '.join(class_distribution_desc),
            'original_label_type': str(y.dtype)
        }
        
        # Add class distribution as separate columns using original labels
        for class_idx, (original_label, count, percentage) in enumerate(zip(unique_classes, class_counts, class_percentages)):
            mapped_label = label_mapping.get(original_label, original_label)
            characteristics[f'class_{mapped_label}_count'] = int(count)
            characteristics[f'class_{mapped_label}_percentage'] = float(percentage)
        
        return characteristics
    
    def generate_detailed_feature_analysis(self, dataset_name: str, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Generate detailed per-feature analysis"""
        try:
            X_float = X.astype(float)
            y_int, label_mapping = self.safe_convert_labels(y)
            
            n_features = X_float.shape[1]
            feature_analysis = []
            
            for feature_idx in range(n_features):
                feature_data = X_float[:, feature_idx]
                
                stats = {
                    'dataset_name': dataset_name,
                    'feature_index': feature_idx,
                    'mean': float(np.mean(feature_data)),
                    'std': float(np.std(feature_data)),
                    'variance': float(np.var(feature_data)),
                    'min': float(np.min(feature_data)),
                    'max': float(np.max(feature_data)),
                    'range': float(np.max(feature_data) - np.min(feature_data)),
                    'coefficient_of_variation': float(np.std(feature_data) / (np.abs(np.mean(feature_data)) + 1e-8)),
                    'q1': float(np.percentile(feature_data, 25)),
                    'median': float(np.percentile(feature_data, 50)),
                    'q3': float(np.percentile(feature_data, 75)),
                    'iqr': float(np.percentile(feature_data, 75) - np.percentile(feature_data, 25)),
                    'is_constant': bool(np.std(feature_data) < 1e-8)
                }
                
                # Add skewness and kurtosis only for non-constant features
                if stats['std'] > 1e-8:
                    stats['skewness'] = float(skew(feature_data))
                    stats['kurtosis'] = float(kurtosis(feature_data))
                else:
                    stats['skewness'] = 0.0
                    stats['kurtosis'] = -3.0  # Constant distribution kurtosis
                
                # Class-conditional statistics
                unique_classes = np.unique(y_int)
                for class_label in unique_classes:
                    class_mask = (y_int == class_label)
                    class_data = feature_data[class_mask]
                    if len(class_data) > 0:
                        stats[f'class_{class_label}_mean'] = float(np.mean(class_data))
                        stats[f'class_{class_label}_std'] = float(np.std(class_data))
                    else:
                        stats[f'class_{class_label}_mean'] = float('nan')
                        stats[f'class_{class_label}_std'] = float('nan')
                
                feature_analysis.append(stats)
            
            return pd.DataFrame(feature_analysis)
            
        except Exception as e:
            logger.error(f"Detailed feature analysis failed for {dataset_name}: {e}")
            return pd.DataFrame()
    
    def generate_class_distribution_analysis(self, dataset_name: str, y: np.ndarray) -> pd.DataFrame:
        """Generate detailed class distribution analysis"""
        try:
            y_int, label_mapping = self.safe_convert_labels(y)
            unique_classes, class_counts = np.unique(y_int, return_counts=True)
            class_percentages = (class_counts / len(y_int)) * 100
            
            class_analysis = []
            for class_label, count, percentage in zip(unique_classes, class_counts, class_percentages):
                original_label = label_mapping.get(class_label, class_label)
                class_analysis.append({
                    'dataset_name': dataset_name,
                    'class_label': original_label,
                    'encoded_class_label': int(class_label),
                    'count': int(count),
                    'percentage': float(percentage),
                    'is_minority': bool(count == np.min(class_counts)),
                    'is_majority': bool(count == np.max(class_counts))
                })
            
            return pd.DataFrame(class_analysis)
            
        except Exception as e:
            logger.error(f"Class distribution analysis failed for {dataset_name}: {e}")
            return pd.DataFrame()
    
    def execute_dataset_characterization(self, output_dir: str = "dataset_characterization"):
        """Execute comprehensive dataset characterization"""
        if not self.datasets:
            self.load_datasets()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        summary_results = []
        all_detailed_features = []
        all_class_distributions = []
        
        for dataset_name, X, y in self.datasets:
            try:
                logger.info(f"Characterizing dataset: {dataset_name}")
                
                # Debug: print original label information
                logger.info(f"  Original labels - dtype: {y.dtype}, unique: {np.unique(y)}")
                
                # Main dataset characterization
                dataset_chars = self.compute_dataset_characteristics(X, y, dataset_name)
                summary_results.append(dataset_chars)
                
                # Detailed feature analysis
                feature_analysis = self.generate_detailed_feature_analysis(dataset_name, X, y)
                if not feature_analysis.empty:
                    all_detailed_features.append(feature_analysis)
                
                # Class distribution analysis
                class_analysis = self.generate_class_distribution_analysis(dataset_name, y)
                if not class_analysis.empty:
                    all_class_distributions.append(class_analysis)
                
                logger.info(f"Completed characterization of {dataset_name}")
                
            except Exception as e:
                logger.error(f"Error characterizing {dataset_name}: {e}")
                continue
        
        # Save all results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Summary results
        if summary_results:
            summary_df = pd.DataFrame(summary_results)
            summary_file = output_path / f"dataset_characterization_summary_{timestamp}.csv"
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Dataset characterization summary saved to {summary_file}")
        
        # Detailed feature analysis
        if all_detailed_features:
            features_df = pd.concat(all_detailed_features, ignore_index=True)
            features_file = output_path / f"detailed_feature_analysis_{timestamp}.csv"
            features_df.to_csv(features_file, index=False)
            logger.info(f"Detailed feature analysis saved to {features_file}")
        
        # Class distribution analysis
        if all_class_distributions:
            classes_df = pd.concat(all_class_distributions, ignore_index=True)
            classes_file = output_path / f"class_distribution_analysis_{timestamp}.csv"
            classes_df.to_csv(classes_file, index=False)
            logger.info(f"Class distribution analysis saved to {classes_file}")
        
        # Generate comprehensive summary table
        if summary_results:
            self.generate_summary_table(summary_df, output_path, timestamp)
        
        return summary_df, features_df, classes_df
    
    def generate_summary_table(self, summary_df: pd.DataFrame, output_path: Path, timestamp: str):
        """Generate a clean summary table of all datasets"""
        
        # Create a simplified summary table for easy reading
        simple_summary = []
        
        for _, row in summary_df.iterrows():
            dataset_summary = {
                'Dataset': row['dataset_name'],
                'Samples': int(row['n_samples']),
                'Features': int(row['n_features']),
                'Classes': int(row['n_classes']),
                'Class Distribution': row['class_distribution_description'],
                'Majority Class': f"{row['majority_class_percentage']:.1f}%",
                'Minority Class': f"{row['minority_class_percentage']:.1f}%",
                'Imbalance Ratio': f"{row['class_imbalance_ratio']:.3f}",
                'Overall Mean': f"{row['overall_mean']:.3f}",
                'Overall Std': f"{row['overall_std']:.3f}",
                'Data Range': f"[{row['data_range_min']:.3f}, {row['data_range_max']:.3f}]",
                'Max Feature Correlation': f"{row['max_feature_correlation']:.3f}",
                'Constant Features': int(row['n_constant_features'])
            }
            simple_summary.append(dataset_summary)
        
        # Create DataFrame and save
        simple_df = pd.DataFrame(simple_summary)
        simple_file = output_path / f"dataset_summary_table_{timestamp}.csv"
        simple_df.to_csv(simple_file, index=False)
        logger.info(f"Simplified summary table saved to {simple_file}")
        
        # Also generate a text report
        self.generate_text_report(simple_df, output_path, timestamp)
        
        return simple_df
    
    def generate_text_report(self, summary_df: pd.DataFrame, output_path: Path, timestamp: str):
        """Generate a text report with dataset summaries"""
        
        report_lines = []
        report_lines.append("DATASET CHARACTERIZATION SUMMARY")
        report_lines.append("=" * 100)
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total datasets characterized: {len(summary_df)}")
        report_lines.append("")
        
        for _, row in summary_df.iterrows():
            report_lines.append(f"DATASET: {row['Dataset']}")
            report_lines.append("-" * 50)
            report_lines.append(f"  Samples: {row['Samples']}")
            report_lines.append(f"  Features: {row['Features']}")
            report_lines.append(f"  Classes: {row['Classes']}")
            report_lines.append(f"  Class Distribution: {row['Class Distribution']}")
            report_lines.append(f"  Majority Class: {row['Majority Class']}")
            report_lines.append(f"  Minority Class: {row['Minority Class']}")
            report_lines.append(f"  Imbalance Ratio: {row['Imbalance Ratio']}")
            report_lines.append(f"  Overall Statistics - Mean: {row['Overall Mean']}, Std: {row['Overall Std']}")
            report_lines.append(f"  Data Range: {row['Data Range']}")
            report_lines.append(f"  Max Feature Correlation: {row['Max Feature Correlation']}")
            report_lines.append(f"  Constant Features: {row['Constant Features']}")
            report_lines.append("")
        
        # Save report
        report_file = output_path / f"dataset_characterization_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Characterization report saved to {report_file}")
        
        # Print to console
        print("\n" + "=" * 100)
        print("DATASET CHARACTERIZATION SUMMARY")
        print("=" * 100)
        for line in report_lines[:min(50, len(report_lines))]:
            print(line)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Dataset Characterization')
    
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    parser.add_argument('--output_dir', type=str, default='dataset_characterization',
                       help='Output directory for characterization results')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    try:
        characterizer = DatasetCharacterizer(config_path=args.config)
        
        logger.info("Starting dataset characterization...")
        summary_df, features_df, classes_df = characterizer.execute_dataset_characterization(
            output_dir=args.output_dir
        )
        
        logger.info("Dataset characterization completed successfully!")
        logger.info(f"Results saved to '{args.output_dir}' directory")
        
        if summary_df is not None:
            print(f"\nCharacterized {len(summary_df)} datasets:")
            for dataset in summary_df['dataset_name']:
                print(f"  - {dataset}")
        
    except Exception as e:
        logger.error(f"Error during dataset characterization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()