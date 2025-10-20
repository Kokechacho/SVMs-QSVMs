import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import Dict, List, Tuple

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResultsVisualizer:
    def __init__(self, results_dir: str = "bayesian_optimization_methodology"):
        self.results_dir = Path(results_dir)
        self.summary_data = None
        self.kernel_mapping = {
            'linear': 'LINEAR',
            'poly': 'POLY', 
            'rbf': 'RBF',
            'custom_hermite': 'HERMITE',
            'custom_gegen': 'GEGEN',
            'custom_alsalam': 'AL-SALAM'
        }
        
        # Mapeo de nombres de métricas en el CSV a los que usamos internamente
        self.metric_mapping = {
            'accuracy': 'accuracy',
            'f1_score': 'f1',  # En el CSV es 'f1_mean'
            'training_time': 'training_time',
            'support_vector_prop': 'support_vector_prop'
        }
        
    def load_latest_results(self):
        """Load the most recent summary results file"""
        summary_files = list(self.results_dir.glob("bayesian_methodology_summary_*.csv"))
        if not summary_files:
            raise FileNotFoundError(f"No summary files found in {self.results_dir}")
        
        # Get the most recent file
        latest_file = max(summary_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading results from: {latest_file}")
        
        self.summary_data = pd.read_csv(latest_file)
        return self.summary_data
    
    def prepare_data_for_table(self, metric: str) -> pd.DataFrame:
        """Prepare data for LaTeX table generation"""
        if self.summary_data is None:
            self.load_latest_results()
        
        # Filter and prepare data
        df = self.summary_data.copy()
        df['kernel_short'] = df['kernel'].map(self.kernel_mapping)
        
        # Mapear la métrica al nombre de columna en el DataFrame
        metric_column_mean = f"{self.metric_mapping[metric]}_mean"
        metric_column_std = f"{self.metric_mapping[metric]}_std"
        
        # Verificar que las columnas existan
        if metric_column_mean not in df.columns or metric_column_std not in df.columns:
            available_columns = ", ".join(df.columns)
            raise KeyError(f"Columnas no encontradas. Esperadas: {metric_column_mean} y {metric_column_std}. Disponibles: {available_columns}")
        
        # Create pivot table
        pivot_data = df.pivot_table(
            index='dataset',
            columns='kernel_short',
            values=[metric_column_mean, metric_column_std],
            aggfunc='first'
        )
        
        return pivot_data
    
    def generate_latex_table(self, metric: str, caption: str, label: str) -> str:
        """Generate LaTeX table for a specific metric"""
        pivot_data = self.prepare_data_for_table(metric)
        
        # Define kernel order for columns
        kernel_order = ['GEGEN', 'RBF', 'POLY', 'AL-SALAM', 'LINEAR', 'HERMITE']
        
        # Start building LaTeX table
        latex_code = []
        latex_code.append("\\begin{table}[htbp]")
        latex_code.append("\\centering")
        latex_code.append(f"\\caption{{{caption}}}")
        latex_code.append(f"\\label{{{label}}}")
        latex_code.append("\\resizebox{\\textwidth}{!}{%")
        latex_code.append("\\begin{tabular}{l" + "c" * len(kernel_order) + "}")
        latex_code.append("\\toprule")
        
        # Header row
        header = "\\textbf{Kernel} & " + " & ".join([f"\\textbf{{{kernel}}}" for kernel in kernel_order]) + " \\\\"
        latex_code.append(header)
        latex_code.append("\\midrule")
        
        # Subheader row
        subheader = "\\textbf{Dataset} & " + " & ".join(["Mean $\\mid$ Std"] * len(kernel_order)) + " \\\\"
        latex_code.append(subheader)
        latex_code.append("\\midrule")
        
        # Process each dataset
        datasets = pivot_data.index.tolist()
        
        for dataset in datasets:
            row_data = []
            dataset_values = []
            
            # Collect values and determine best
            for kernel in kernel_order:
                mean_key = (f'{self.metric_mapping[metric]}_mean', kernel)
                std_key = (f'{self.metric_mapping[metric]}_std', kernel)
                
                if mean_key in pivot_data.columns and std_key in pivot_data.columns:
                    mean_val = pivot_data[mean_key].loc[dataset]
                    std_val = pivot_data[std_key].loc[dataset]
                    
                    # Check for NaN
                    if pd.isna(mean_val) or pd.isna(std_val):
                        dataset_values.append((kernel, np.nan, np.nan))
                    else:
                        if metric in ['accuracy', 'f1_score']:
                            # Convert to percentage for accuracy and f1_score
                            mean_val = mean_val * 100
                            std_val = std_val * 100
                        
                        dataset_values.append((kernel, mean_val, std_val))
                else:
                    dataset_values.append((kernel, np.nan, np.nan))
            
            # Determine best value
            valid_values = [(k, m, s) for k, m, s in dataset_values if not np.isnan(m)]
            if not valid_values:
                # No valid values for this dataset, skip ranking and best
                best_kernel = None
            else:
                if metric in ['accuracy', 'f1_score']:
                    # Higher is better
                    best_kernel = max(valid_values, key=lambda x: x[1])[0]
                else:
                    # Lower is better for training_time and support_vector_prop
                    best_kernel = min(valid_values, key=lambda x: x[1])[0]
            
            # Build row content
            for kernel, mean_val, std_val in dataset_values:
                if np.isnan(mean_val) or np.isnan(std_val):
                    cell_content = "N/A"
                else:
                    # Format based on metric type
                    if metric in ['accuracy', 'f1_score']:
                        mean_fmt = f"{mean_val:.2f}"
                        std_fmt = f"{std_val:.2f}"
                    elif metric == 'training_time':
                        mean_fmt = f"{mean_val:.4f}"
                        std_fmt = f"{std_val:.4f}"
                    else:  # support_vector_prop
                        mean_fmt = f"{mean_val:.4f}"
                        std_fmt = f"{std_val:.4f}"
                    
                    # Determine ranking
                    ranked_values = sorted([(k, m) for k, m, s in valid_values], 
                                         key=lambda x: x[1], reverse=(metric in ['accuracy', 'f1_score']))
                    ranking = next((i+1 for i, (k, m) in enumerate(ranked_values) if k == kernel), None)
                    
                    if kernel == best_kernel:
                        cell_content = f"\\textbf{{{mean_fmt} $\\pm$ {std_fmt}}}\\\\{ranking}"
                    else:
                        cell_content = f"{mean_fmt} $\\pm$ {std_fmt}\\\\{ranking}"
                
                row_data.append(f"\\begin{{tabular}}{{@{{}}c@{{}}}}{cell_content}\\end{{tabular}}")
            
            # Add dataset row
            row = f"{dataset} & " + " & ".join(row_data) + " \\\\"
            latex_code.append(row)
            latex_code.append("\\hline")
        
        # Calculate average ranks
        latex_code.append("\\textbf{Avg. Rank} & ")
        avg_ranks = []
        
        for kernel in kernel_order:
            kernel_ranks = []
            for dataset in datasets:
                dataset_values = []
                for k in kernel_order:
                    mean_key = (f'{self.metric_mapping[metric]}_mean', k)
                    if mean_key in pivot_data.columns:
                        mean_val = pivot_data[mean_key].loc[dataset]
                        if not pd.isna(mean_val):
                            dataset_values.append((k, mean_val))
                
                if dataset_values:
                    ranked = sorted(dataset_values, key=lambda x: x[1], reverse=(metric in ['accuracy', 'f1_score']))
                    rank = next((i+1 for i, (k, m) in enumerate(ranked) if k == kernel), None)
                    if rank is not None:
                        kernel_ranks.append(rank)
            
            if kernel_ranks:
                avg_rank = np.mean(kernel_ranks)
                # Encontrar el mejor promedio (menor valor)
                min_avg = min(avg_ranks) if avg_ranks else avg_rank
                avg_ranks.append(avg_rank)
                formatted_avg = f"\\textbf{{{avg_rank:.2f}}}" if avg_rank == min_avg else f"{avg_rank:.2f}"
            else:
                formatted_avg = "N/A"
                avg_ranks.append(np.nan)  # Para mantener la lista de promedios, aunque no se use en la comparación
        
        # Ahora formateamos los promedios, teniendo en cuenta que algunos pueden ser NaN
        formatted_avgs = []
        min_avg = min([x for x in avg_ranks if not np.isnan(x)])
        for avg_rank in avg_ranks:
            if np.isnan(avg_rank):
                formatted_avgs.append("N/A")
            else:
                if avg_rank == min_avg:
                    formatted_avgs.append(f"\\textbf{{{avg_rank:.2f}}}")
                else:
                    formatted_avgs.append(f"{avg_rank:.2f}")
        
        latex_code.append(" & ".join(formatted_avgs) + " \\\\")
        
        # End table
        latex_code.append("\\bottomrule")
        latex_code.append("\\end{tabular}%")
        latex_code.append("}")
        latex_code.append("\\end{table}")
        
        return "\n".join(latex_code)
    
    def generate_all_tables(self, output_file: str = "results_tables.tex"):
        """Generate all LaTeX tables and save to file"""
        
        metric_configs = [
            {
                'metric': 'accuracy',
                'caption': 'Statistics on kernel performance according to classification accuracy',
                'label': 'tab:accuracy_results'
            },
            {
                'metric': 'f1_score', 
                'caption': 'Statistics on kernel performance according to F1-score',
                'label': 'tab:f1_results'
            },
            {
                'metric': 'training_time',
                'caption': 'Statistics on kernel performance according to training time (seconds)',
                'label': 'tab:training_time_results'
            },
            {
                'metric': 'support_vector_prop',
                'caption': 'Statistics on kernel performance according to support vector proportion',
                'label': 'tab:support_vector_results'
            }
        ]
        
        all_tables = []
        
        for config in metric_configs:
            logger.info(f"Generating table for {config['metric']}")
            try:
                table = self.generate_latex_table(
                    metric=config['metric'],
                    caption=config['caption'],
                    label=config['label']
                )
                all_tables.append(table)
                all_tables.append("")  # Add empty line between tables
            except Exception as e:
                logger.error(f"Error generating table for {config['metric']}: {e}")
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(all_tables))
        
        logger.info(f"All tables saved to {output_file}")
        
        # Also print to console
        print("\n" + "="*80)
        print("GENERATED LaTeX TABLES")
        print("="*80)
        for table in all_tables:
            print(table)
        
        return all_tables

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate LaTeX tables from Bayesian Optimization results')
    
    parser.add_argument('--results_dir', type=str, default='bayesian_optimization_methodology',
                       help='Directory containing results files (default: bayesian_optimization_methodology)')
    parser.add_argument('--output_file', type=str, default='results_tables.tex',
                       help='Output LaTeX file (default: results_tables.tex)')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    try:
        visualizer = ResultsVisualizer(results_dir=args.results_dir)
        visualizer.load_latest_results()
        
        logger.info("Generating LaTeX tables from results...")
        tables = visualizer.generate_all_tables(output_file=args.output_file)
        
        logger.info("Table generation completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"Results not found: {e}")
    except Exception as e:
        logger.error(f"Error generating tables: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()