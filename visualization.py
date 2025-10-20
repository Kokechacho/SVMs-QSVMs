import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AcademicBoxPlotGenerator:
    def __init__(self, results_dir: str = "bayesian_optimization_methodology"):
        self.results_dir = Path(results_dir)
        self.detailed_data = None
        self.summary_data = None

        # Academic style configuration
        self.set_academic_style()

        # Kernel mapping and colors
        self.kernel_mapping = {
            'linear': 'LINEAR',
            'poly': 'POLY', 
            'rbf': 'RBF',
            'custom_hermite': 'HERMITE',
            'custom_gegen': 'GEGEN',
            'custom_alsalam': 'AL-SALAM'
        }

        # ensure the ordering here is the canonical order used in plots
        self.kernel_order = ['LINEAR', 'POLY', 'RBF', 'HERMITE', 'GEGEN', 'AL-SALAM']

        self.kernel_colors = {
            'LINEAR': '#1f77b4',
            'POLY': '#ff7f0e',
            'RBF': '#2ca02c',
            'HERMITE': '#d62728',
            'GEGEN': '#9467bd',
            'AL-SALAM': '#8c564b'
        }

    def set_academic_style(self):
        """Set academic plotting style"""
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'mathtext.fontset': 'stix',
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })

    def load_latest_results(self):
        """Load the most recent detailed results file"""
        detailed_files = list(self.results_dir.glob("bayesian_methodology_detailed_*.csv"))
        summary_files = list(self.results_dir.glob("bayesian_methodology_summary_*.csv"))

        if not detailed_files:
            raise FileNotFoundError(f"No detailed results files found in {self.results_dir}")

        # Get the most recent files
        latest_detailed = max(detailed_files, key=lambda x: x.stat().st_mtime)
        latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime) if summary_files else None

        logger.info(f"Loading detailed results from: {latest_detailed}")
        self.detailed_data = pd.read_csv(latest_detailed)

        if latest_summary:
            logger.info(f"Loading summary results from: {latest_summary}")
            self.summary_data = pd.read_csv(latest_summary)

        # Map kernel names (safely)
        self.detailed_data['kernel'] = self.detailed_data['kernel'].map(self.kernel_mapping).fillna(self.detailed_data['kernel'])

        # Convert accuracy and f1_score to percentages for better readability if present
        if 'accuracy' in self.detailed_data.columns:
            self.detailed_data['accuracy_percent'] = self.detailed_data['accuracy'] * 100
        if 'f1_score' in self.detailed_data.columns:
            self.detailed_data['f1_score_percent'] = self.detailed_data['f1_score'] * 100

        logger.info(f"Loaded data for {len(self.detailed_data['dataset'].unique())} datasets")
        logger.info(f"Available kernels: {self.detailed_data['kernel'].unique()}")

        return self.detailed_data

    # ----------------------------- New plotting strategies -----------------------------
    def create_per_dataset_plots(self, output_dir: str = "boxplot_results/per_dataset"):
        """Create one consolidated figure per dataset with subplots for each metric.
        This keeps each dataset on a separate image so the six kernels are easy to compare.
        """
        if self.detailed_data is None:
            self.load_latest_results()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        datasets = sorted(self.detailed_data['dataset'].unique())
        metrics = [
            ('accuracy_percent', 'Accuracy (%)', False),
            ('f1_score_percent', 'F1-Score (%)', False),
            ('training_time', 'Training time (s)', True),
            ('support_vector_prop', 'Support vector proportion', False)
        ]

        generated = []
        for dataset in datasets:
            dataset_df = self.detailed_data[self.detailed_data['dataset'] == dataset]
            if dataset_df.empty:
                continue

            # Create a tall figure: one row per metric
            n_metrics = len(metrics)
            fig, axes = plt.subplots(n_metrics, 1, figsize=(7.5, 3.5 * n_metrics), constrained_layout=True)

            for ax, (col, ylabel, log_scale) in zip(axes, metrics):
                if col not in dataset_df.columns:
                    ax.set_visible(False)
                    continue

                # Order kernels to keep consistency
                order = [k for k in self.kernel_order if k in dataset_df['kernel'].unique()]

                sns.boxplot(
                    data=dataset_df,
                    x='kernel',
                    y=col,
                    ax=ax,
                    order=order,
                    palette=[self.kernel_colors[k] for k in order],
                    width=0.6,
                    linewidth=0.9
                )

                sns.stripplot(
                    data=dataset_df,
                    x='kernel',
                    y=col,
                    ax=ax,
                    order=order,
                    color='black',
                    alpha=0.5,
                    size=3,
                    jitter=0.18
                )

                ax.set_title(f"{dataset} — {ylabel}", fontweight='bold')
                ax.set_xlabel('' if ax is not axes[-1] else 'Kernel')
                ax.set_ylabel(ylabel)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
                ax.grid(True, alpha=0.25)

                if log_scale:
                    # some values might be zero — offset by a small epsilon if needed
                    ax.set_yscale('log')

            # Save figure per dataset
            filename = out / f"{dataset.replace(' ', '_')}_metrics.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            generated.append(filename)
            logger.info(f"Saved per-dataset plot: {filename}")

        return generated

    def create_combined_summary_plots(self, output_dir: str = "boxplot_results/summary"):
        """Create combined summary plots (one per metric) that show kernel mean +/- std across datasets.
        These are useful to see global trends without having every dataset jammed into one axis.
        """
        if self.detailed_data is None:
            self.load_latest_results()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        metrics = [
            ('accuracy_percent', 'Accuracy (%)'),
            ('f1_score_percent', 'F1-Score (%)'),
            ('training_time', 'Training time (s)'),
            ('support_vector_prop', 'Support vector proportion')
        ]

        generated = []
        for col, ylabel in metrics:
            if col not in self.detailed_data.columns:
                logger.warning(f"Metric {col} not in detailed data — skipping summary plot")
                continue

            # compute mean and std per kernel across datasets
            grouped = self.detailed_data.groupby('kernel')[col].agg(['mean', 'std', 'count']).reindex(self.kernel_order)
            grouped = grouped.dropna(how='all')

            # Keep only kernels with data
            grouped = grouped[grouped['count'] > 0]
            if grouped.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))

            x = np.arange(len(grouped))
            means = grouped['mean'].values
            stds = grouped['std'].values
            labels = grouped.index.tolist()
            colors = [self.kernel_colors.get(k, '#333333') for k in labels]

            ax.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor='black', alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha='right')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(f"Kernel mean ± std across datasets — {ylabel}", fontweight='bold')
            ax.grid(True, alpha=0.25)

            filename = out / f"summary_{col}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            generated.append(filename)
            logger.info(f"Saved summary plot: {filename}")

        return generated

    # Keep the old combined function but make it call the new ones depending on mode
    def create_combined_kernel_comparison(self, output_dir: str = "boxplot_results", mode: str = 'both'):
        """Create combined comparison plots for all metrics. mode: 'per_dataset', 'summary', or 'both'"""
        if self.detailed_data is None:
            self.load_latest_results()

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        generated_files = []
        if mode in ('per_dataset', 'both'):
            logger.info("Generating per-dataset plots (one image per dataset with all metrics)")
            generated_files.extend(self.create_per_dataset_plots(output_dir=str(output_path / 'per_dataset')))

        if mode in ('summary', 'both'):
            logger.info("Generating combined summary plots (one image per metric aggregated across datasets)")
            generated_files.extend(self.create_combined_summary_plots(output_dir=str(output_path / 'summary')))

        return generated_files

    def create_kernel_rank_plot(self, output_dir: str = "boxplot_results"):
        """Create a plot showing kernel rankings across datasets"""
        if self.summary_data is None:
            # try to load summary if not present
            self.load_latest_results()

        if self.summary_data is None:
            logger.warning("Summary data not available for ranking plot")
            return None

        # Calculate rankings for each dataset and metric
        datasets = self.summary_data['dataset'].unique()
        kernels = self.summary_data['kernel'].unique()

        # Prepare ranking data
        ranking_data = []

        for dataset in datasets:
            dataset_data = self.summary_data[self.summary_data['dataset'] == dataset]

            # Rank by accuracy (higher is better) if present
            if 'accuracy_mean' in dataset_data.columns:
                accuracy_rank = dataset_data['accuracy_mean'].rank(ascending=False)
            else:
                accuracy_rank = pd.Series(np.nan, index=dataset_data.index)

            # Rank by training time (lower is better)
            if 'training_time_mean' in dataset_data.columns:
                time_rank = dataset_data['training_time_mean'].rank(ascending=True)
            else:
                time_rank = pd.Series(np.nan, index=dataset_data.index)

            # Rank by support vectors (lower is better)
            if 'support_vector_prop_mean' in dataset_data.columns:
                sv_rank = dataset_data['support_vector_prop_mean'].rank(ascending=True)
            else:
                sv_rank = pd.Series(np.nan, index=dataset_data.index)

            for idx_row, row in dataset_data.iterrows():
                kernel = row['kernel']
                ranking_data.append({
                    'dataset': dataset,
                    'kernel': kernel,
                    'accuracy_rank': accuracy_rank.get(idx_row, np.nan),
                    'training_time_rank': time_rank.get(idx_row, np.nan),
                    'support_vector_rank': sv_rank.get(idx_row, np.nan),
                    'overall_rank': np.nanmean([
                        accuracy_rank.get(idx_row, np.nan),
                        time_rank.get(idx_row, np.nan),
                        sv_rank.get(idx_row, np.nan)
                    ])
                })

        ranking_df = pd.DataFrame(ranking_data)

        # Create ranking plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        metrics = ['accuracy_rank', 'training_time_rank', 'support_vector_rank', 'overall_rank']
        titles = ['Accuracy Ranking', 'Training Time Ranking', 'Support Vector Ranking', 'Overall Ranking']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]

            # Pivot for boxplot
            pivot_data = ranking_df.pivot_table(
                index='dataset', 
                columns='kernel', 
                values=metric
            )

            # Reorder columns by median rank
            median_ranks = pivot_data.median().sort_values()
            pivot_data = pivot_data[median_ranks.index]

            # Create boxplot
            sns.boxplot(data=pivot_data, ax=ax, palette=[self.kernel_colors.get(k, '#777777') for k in pivot_data.columns], width=0.7)

            # Customize
            ax.set_title(title, fontweight='bold', pad=10)
            ax.set_xlabel('Kernel', fontweight='bold')
            ax.set_ylabel('Rank (lower is better)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            # Rotate x-axis labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        fig.suptitle('Kernel Performance Rankings Across All Datasets', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        # Save plot
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        output_file = output_path / "kernel_rankings.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ranking plot to {output_file}")

        plt.close(fig)
        return output_file


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate academic boxplots from Bayesian Optimization results')

    parser.add_argument('--results_dir', type=str, default='bayesian_optimization_methodology',
                       help='Directory containing results files')
    parser.add_argument('--output_dir', type=str, default='boxplot_results',
                       help='Output directory for plots')
    parser.add_argument('--include_rankings', action='store_true',
                       help='Include kernel ranking plots')
    parser.add_argument('--mode', type=str, choices=['per_dataset', 'summary', 'both'], default='both',
                       help='Which plotting mode to use: per_dataset (one image per dataset), summary (one image per metric aggregated), or both')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()

    try:
        # Initialize plot generator
        plotter = AcademicBoxPlotGenerator(results_dir=args.results_dir)

        # Load data
        plotter.load_latest_results()

        # Generate comparison plots
        logger.info("Generating metric comparison plots...")
        plot_files = plotter.create_combined_kernel_comparison(output_dir=args.output_dir, mode=args.mode)

        # Generate ranking plot if requested
        if args.include_rankings:
            logger.info("Generating kernel ranking plot...")
            rank_file = plotter.create_kernel_rank_plot(output_dir=args.output_dir)
            if rank_file:
                plot_files.append(rank_file)

        logger.info(f"Generated {len(plot_files)} plots in '{args.output_dir}' directory")
        logger.info("Plot generation completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"Results not found: {e}")
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
