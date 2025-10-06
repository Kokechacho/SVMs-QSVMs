import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import scipy.stats as stats
from typing import Dict, List, Tuple
import logging

# Configuración de logging y estilo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('default')
sns.set_palette("husl")

class ResultsAnalyzer:
    def __init__(self, summary_path: str = "statistical_results/statistical_summary.csv", 
                 detailed_path: str = "statistical_results/detailed_results.csv"):
        self.summary_path = Path(summary_path)
        self.detailed_path = Path(detailed_path)
        self.summary_df = None
        self.detailed_df = None
        self.rankings_df = None
        
    def load_data(self):
        """Carga los datos de resultados"""
        try:
            self.summary_df = pd.read_csv(self.summary_path)
            # Convertir tipos de datos numéricos
            self.summary_df = self._convert_dtypes(self.summary_df)
            logger.info(f"📊 Cargados {len(self.summary_df)} resultados del resumen")
            
            if self.detailed_path.exists():
                self.detailed_df = pd.read_csv(self.detailed_path)
                self.detailed_df = self._convert_dtypes(self.detailed_df)
                logger.info(f"📈 Cargados {len(self.detailed_df)} resultados detallados")
            else:
                logger.warning("⚠️ No se encontró archivo de resultados detallados")
                
            return True
        except Exception as e:
            logger.error(f"❌ Error cargando datos: {e}")
            return False
    
    def _convert_dtypes(self, df):
        """Convierte tipos de datos numpy a Python nativos para evitar problemas de serialización"""
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                # Intentar convertir columnas que podrían ser numéricas
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        return df
    
    def _convert_to_python_types(self, obj):
        """Convierte numpy types a Python types para JSON serialization"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_python_types(item) for item in obj]
        else:
            return obj
    
    def calculate_rankings(self):
        """Calcula rankings por dataset y rankings globales"""
        if self.summary_df is None:
            logger.error("❌ No hay datos cargados")
            return
        
        # Calcular ranking por dataset
        rankings = []
        for dataset in self.summary_df['dataset'].unique():
            dataset_data = self.summary_df[self.summary_df['dataset'] == dataset].copy()
            # Ordenar por accuracy_mean descendente y asignar ranking
            dataset_data = dataset_data.sort_values('accuracy_mean', ascending=False)
            dataset_data['ranking'] = range(1, len(dataset_data) + 1)
            rankings.append(dataset_data)
        
        self.rankings_df = pd.concat(rankings, ignore_index=True)
        
        # Calcular estadísticas de ranking por kernel
        kernel_rankings = self.rankings_df.groupby('kernel').agg({
            'ranking': ['mean', 'std', 'min', 'max'],
            'accuracy_mean': ['mean', 'std'],
            'dataset': 'count'  # Número de datasets donde aparece
        }).round(3)
        
        # Aplanar columnas multiindex
        kernel_rankings.columns = ['_'.join(col).strip() for col in kernel_rankings.columns.values]
        kernel_rankings = kernel_rankings.rename(columns={
            'ranking_mean': 'rank_mean',
            'ranking_std': 'rank_std',
            'ranking_min': 'rank_min',
            'ranking_max': 'rank_max',
            'accuracy_mean_mean': 'accuracy_global_mean',
            'accuracy_mean_std': 'accuracy_global_std',
            'dataset_count': 'n_datasets'
        })
        
        kernel_rankings = kernel_rankings.sort_values('rank_mean')
        
        return kernel_rankings
    
    def create_comprehensive_analysis(self):
        """Crea un análisis completo con múltiples visualizaciones"""
        if not self.load_data():
            return
        
        # Crear directorio de resultados
        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        # Calcular rankings
        kernel_rankings = self.calculate_rankings()
        
        # 1. Gráfico de rankings por dataset
        self._plot_rankings_heatmap(output_dir)
        
        # 2. Gráfico de precisión por kernel y dataset
        self._plot_accuracy_by_kernel_dataset(output_dir)
        
        # 3. Gráfico de comparación global de kernels
        self._plot_global_comparison(output_dir)
        
        # 4. Gráfico de distribución de accuracy por kernel
        if self.detailed_df is not None:
            self._plot_accuracy_distribution(output_dir)
        
        # 5. Gráfico de tiempos de ejecución
        self._plot_execution_times(output_dir)
        
        # 6. Gráfico de rankings promedio
        self._plot_average_rankings(output_dir, kernel_rankings)
        
        # 7. Análisis estadístico de diferencias
        self._perform_statistical_analysis(output_dir)
        
        # 8. Generar reporte completo
        self._generate_comprehensive_report(output_dir, kernel_rankings)
        
        logger.info(f"✅ Análisis completado. Resultados guardados en {output_dir}/")
    
    def _plot_rankings_heatmap(self, output_dir: Path):
        """Crea heatmap de rankings por dataset"""
        plt.figure(figsize=(12, 8))
        
        # Crear matriz de rankings
        pivot_rankings = self.rankings_df.pivot_table(
            index='dataset', 
            columns='kernel', 
            values='ranking',
            aggfunc='first'
        )
        
        # Ordenar kernels por ranking promedio
        kernel_order = self.rankings_df.groupby('kernel')['ranking'].mean().sort_values().index
        pivot_rankings = pivot_rankings[kernel_order]
        
        sns.heatmap(
            pivot_rankings,
            annot=True,
            cmap='RdYlGn_r',  # Rojo para mal ranking, verde para bueno
            fmt='d',
            cbar_kws={'label': 'Ranking (1 = mejor)'},
            linewidths=0.5
        )
        
        plt.title('Ranking de Kernels por Dataset\n(1 = Mejor Rendimiento)', fontsize=14, fontweight='bold')
        plt.xlabel('Kernel')
        plt.ylabel('Dataset')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'rankings_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_accuracy_by_kernel_dataset(self, output_dir: Path):
        """Crea gráfico de precisión por kernel y dataset"""
        plt.figure(figsize=(14, 8))
        
        # Ordenar kernels por accuracy global
        kernel_order = self.summary_df.groupby('kernel')['accuracy_mean'].mean().sort_values(ascending=False).index
        
        ax = sns.barplot(
            data=self.summary_df,
            x='dataset',
            y='accuracy_mean',
            hue='kernel',
            hue_order=kernel_order
        )
        
        # Añadir valores en las barras
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3, fontsize=8)
        
        plt.title('Precisión Media por Dataset y Kernel', fontsize=14, fontweight='bold')
        plt.xlabel('Dataset')
        plt.ylabel('Precisión Media')
        plt.legend(title='Kernel', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_by_dataset_kernel.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_global_comparison(self, output_dir: Path):
        """Crea gráfico de comparación global de kernels"""
        plt.figure(figsize=(12, 8))
        
        # Calcular estadísticas globales por kernel
        global_stats = self.summary_df.groupby('kernel').agg({
            'accuracy_mean': ['mean', 'std', 'count'],
            'time_mean': 'mean'
        }).round(3)
        
        global_stats.columns = ['accuracy_mean', 'accuracy_std', 'n_datasets', 'time_mean']
        global_stats = global_stats.sort_values('accuracy_mean', ascending=False)
        
        # Gráfico de precisión global
        plt.subplot(2, 1, 1)
        bars = plt.bar(global_stats.index, global_stats['accuracy_mean'], 
                      yerr=global_stats['accuracy_std'], capsize=5, alpha=0.7)
        plt.title('Precisión Media Global por Kernel', fontweight='bold')
        plt.ylabel('Precisión Media')
        plt.xticks(rotation=45)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, global_stats['accuracy_mean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Gráfico de tiempos de ejecución
        plt.subplot(2, 1, 2)
        bars = plt.bar(global_stats.index, global_stats['time_mean'], alpha=0.7, color='orange')
        plt.title('Tiempo Medio de Ejecución por Kernel', fontweight='bold')
        plt.ylabel('Tiempo (segundos)')
        plt.xlabel('Kernel')
        plt.xticks(rotation=45)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, global_stats['time_mean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'global_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_accuracy_distribution(self, output_dir: Path):
        """Crea gráfico de distribución de accuracy por kernel"""
        plt.figure(figsize=(14, 8))
        
        # Ordenar kernels por mediana de accuracy
        kernel_order = self.detailed_df.groupby('kernel')['accuracy'].median().sort_values(ascending=False).index
        
        sns.boxplot(
            data=self.detailed_df,
            x='kernel',
            y='accuracy',
            order=kernel_order,
            showfliers=False
        )
        
        plt.title('Distribución de Precisión por Kernel (35 ejecuciones)', fontsize=14, fontweight='bold')
        plt.xlabel('Kernel')
        plt.ylabel('Precisión')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Violin plot para mejor visualización de distribuciones
        plt.figure(figsize=(14, 8))
        sns.violinplot(
            data=self.detailed_df,
            x='kernel',
            y='accuracy',
            order=kernel_order,
            inner='quartile'
        )
        plt.title('Distribución de Precisión - Violin Plot', fontsize=14, fontweight='bold')
        plt.xlabel('Kernel')
        plt.ylabel('Precisión')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_violin.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_execution_times(self, output_dir: Path):
        """Crea gráfico de tiempos de ejecución"""
        if self.detailed_df is None:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Ordenar kernels por tiempo medio
        kernel_order = self.detailed_df.groupby('kernel')['train_time'].mean().sort_values(ascending=False).index
        
        sns.boxplot(
            data=self.detailed_df,
            x='kernel',
            y='train_time',
            order=kernel_order,
            showfliers=False
        )
        
        plt.title('Distribución de Tiempos de Ejecución por Kernel', fontsize=14, fontweight='bold')
        plt.xlabel('Kernel')
        plt.ylabel('Tiempo (segundos)')
        plt.xticks(rotation=45)
        plt.yscale('log')  # Escala logarítmica para mejor visualización
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'execution_times.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_average_rankings(self, output_dir: Path, kernel_rankings):
        """Crea gráfico de rankings promedio"""
        plt.figure(figsize=(10, 6))
        
        colors = plt.cm.RdYlGn_r((kernel_rankings['rank_mean'] - 1) / 
                                (len(kernel_rankings) - 1))
        
        bars = plt.bar(kernel_rankings.index, kernel_rankings['rank_mean'],
                      color=colors, alpha=0.7, yerr=kernel_rankings['rank_std'], capsize=5)
        
        plt.title('Ranking Promedio de Kernels\n(1 = Mejor, menor = mejor)', fontsize=14, fontweight='bold')
        plt.xlabel('Kernel')
        plt.ylabel('Ranking Promedio')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, kernel_rankings['rank_mean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'average_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _perform_statistical_analysis(self, output_dir: Path):
        """Realiza análisis estadístico de diferencias entre kernels"""
        if self.detailed_df is None:
            return
            
        from scipy.stats import friedmanchisquare, kruskal
        
        # Preparar datos para Friedman test (por dataset)
        datasets = self.detailed_df['dataset'].unique()
        kernels = self.detailed_df['kernel'].unique()
        
        results = []
        
        for dataset in datasets:
            dataset_data = []
            for kernel in kernels:
                kernel_data = self.detailed_df[
                    (self.detailed_df['dataset'] == dataset) & 
                    (self.detailed_df['kernel'] == kernel)
                ]['accuracy'].values
                
                if len(kernel_data) > 0:
                    dataset_data.append(kernel_data[:35])  # Tomar máximo 35 muestras
                else:
                    dataset_data.append([])
            
            # Friedman test por dataset
            try:
                # Filtrar kernels que tienen datos
                valid_data = [data for data in dataset_data if len(data) > 0]
                if len(valid_data) < 2:  # Necesitamos al menos 2 grupos
                    continue
                    
                friedman_stat, friedman_p = friedmanchisquare(*valid_data)
                
                # Kruskal-Wallis test
                kruskal_stat, kruskal_p = kruskal(*valid_data)
                
                results.append({
                    'dataset': dataset,
                    'friedman_statistic': float(friedman_stat),
                    'friedman_pvalue': float(friedman_p),
                    'kruskal_statistic': float(kruskal_stat),
                    'kruskal_pvalue': float(kruskal_p),
                    'significant_friedman': friedman_p < 0.05,
                    'significant_kruskal': kruskal_p < 0.05
                })
            except Exception as e:
                logger.warning(f"⚠️ Error en test estadístico para {dataset}: {e}")
                continue
        
        if results:
            stats_df = pd.DataFrame(results)
            stats_df.to_csv(output_dir / 'statistical_tests.csv', index=False)
            
            # Resumen de tests estadísticos - asegurar tipos nativos de Python
            summary_stats = {
                'n_datasets': int(len(stats_df)),
                'n_significant_friedman': int(stats_df['significant_friedman'].sum()),
                'n_significant_kruskal': int(stats_df['significant_kruskal'].sum()),
                'mean_friedman_p': float(stats_df['friedman_pvalue'].mean()),
                'mean_kruskal_p': float(stats_df['kruskal_pvalue'].mean())
            }
            
            # Convertir a tipos Python nativos
            summary_stats = self._convert_to_python_types(summary_stats)
            
            with open(output_dir / 'statistical_tests_summary.json', 'w') as f:
                json.dump(summary_stats, f, indent=2)
        else:
            logger.warning("⚠️ No se pudieron realizar tests estadísticos")
    
    def _generate_comprehensive_report(self, output_dir: Path, kernel_rankings):
        """Genera un reporte completo en texto"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("📊 REPORTE COMPLETO DE ANÁLISIS DE RESULTADOS")
        report_lines.append("=" * 80)
        report_lines.append(f"Total de datasets analizados: {len(self.summary_df['dataset'].unique())}")
        report_lines.append(f"Total de kernels analizados: {len(self.summary_df['kernel'].unique())}")
        report_lines.append(f"Total de ejecuciones: {len(self.detailed_df) if self.detailed_df is not None else 'N/A'}")
        report_lines.append("")
        
        # Mejores kernels por ranking
        report_lines.append("🥇 TOP KERNELS POR RANKING PROMEDIO")
        report_lines.append("-" * 50)
        for i, (kernel, row) in enumerate(kernel_rankings.iterrows(), 1):
            report_lines.append(
                f"{i:2d}. {kernel:12} | Ranking: {row['rank_mean']:.2f} ± {row['rank_std']:.2f} | "
                f"Precisión: {row['accuracy_global_mean']:.2f} | "
                f"Datasets: {int(row['n_datasets'])}"
            )
        
        report_lines.append("")
        
        # Mejor kernel por dataset
        report_lines.append("🏆 MEJOR KERNEL POR DATASET")
        report_lines.append("-" * 50)
        best_by_dataset = self.rankings_df[self.rankings_df['ranking'] == 1]
        for _, row in best_by_dataset.iterrows():
            report_lines.append(f"{row['dataset']:15} → {row['kernel']:12} ({row['accuracy_mean']:.2f})")
        
        report_lines.append("")
        
        # Análisis de consistencia
        report_lines.append("📈 ANÁLISIS DE CONSISTENCIA")
        report_lines.append("-" * 50)
        consistency = self.rankings_df.groupby('kernel').agg({
            'ranking': ['std', 'mean'],
            'accuracy_std': 'mean'
        }).round(3)
        
        consistency.columns = ['std_ranking', 'mean_ranking', 'mean_accuracy_std']
        consistency = consistency.sort_values('std_ranking')
        
        for kernel, row in consistency.iterrows():
            report_lines.append(
                f"{kernel:12} | Std Ranking: {row['std_ranking']:.2f} | "
                f"Std Precisión: {row['mean_accuracy_std']:.3f}"
            )
        
        # Ganadores por dataset
        report_lines.append("")
        report_lines.append("📊 DISTRIBUCIÓN DE VICTORIAS POR KERNEL")
        report_lines.append("-" * 50)
        wins_by_kernel = best_by_dataset['kernel'].value_counts()
        for kernel, wins in wins_by_kernel.items():
            report_lines.append(f"{kernel:12} → {wins:2d} datasets ganados")
        
        # Guardar reporte
        with open(output_dir / 'comprehensive_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Guardar rankings en CSV
        kernel_rankings.to_csv(output_dir / 'kernel_rankings_detailed.csv')
        
        logger.info("📄 Reporte de análisis generado")

def main():
    """Función principal"""
    analyzer = ResultsAnalyzer()
    
    logger.info("🔍 INICIANDO ANÁLISIS COMPLETO DE RESULTADOS")
    logger.info("📈 Generando gráficos y estadísticas...")
    
    analyzer.create_comprehensive_analysis()
    
    logger.info("✅ Análisis completado!")
    logger.info("📊 Resultados guardados en directorio 'analysis_results/'")

if __name__ == "__main__":
    main()