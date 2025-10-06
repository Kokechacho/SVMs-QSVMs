import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
from Data.loader import load_all_datasets
from Kernels.all_kernels import build_single_kernel
from Model.train import train_svm
import scipy.stats as stats
import time
import concurrent.futures

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TIMEOUT = 120  

class StatisticalEvaluator:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.load_base_config()
        self.load_datasets()
        
    def load_base_config(self):
        """Carga la configuración base"""
        with open(self.config_path, 'r') as f:
            self.base_config = json.load(f)
    
    def load_datasets(self):
        """Carga todos los datasets"""
        data_cfg = self.base_config['data']
        self.datasets = load_all_datasets(
            uci_list=data_cfg['uci'],
            libsvm_dir=Path(data_cfg['libsvm_dir']),
            libsvm_files=data_cfg['libsvm']
        )
        logger.info(f"📊 Cargados {len(self.datasets)} datasets")
    
    def load_optimized_parameters(self, optimization_results_path: str = "optimization_results/best_hyperparameters.json"):
        """Carga los parámetros optimizados y normaliza los tipos de datos"""
        try:
            with open(optimization_results_path, 'r') as f:
                params = json.load(f)
            
            # Normalizar tipos de datos
            return self._normalize_parameter_types(params)
            
        except FileNotFoundError:
            logger.error(f"❌ No se encontraron parámetros optimizados en {optimization_results_path}")
            return {}
    
    def _normalize_parameter_types(self, params_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Normaliza los tipos de datos de los parámetros y unifica nombres"""
        normalized_params = {}
        
        for key, params in params_dict.items():
            normalized_params[key] = {}
            
            for param_name, param_value in params.items():
                # Normalizar parámetros numéricos
                if param_name in ['C', 'gamma_custom', 'gamma_value', 'coef0', 'alpha', 'q']:
                    try:
                        if isinstance(param_value, str):
                            normalized_params[key][param_name] = float(param_value)
                        else:
                            normalized_params[key][param_name] = param_value
                    except (ValueError, TypeError):
                        normalized_params[key][param_name] = param_value
                        logger.warning(f"⚠️ No se pudo convertir {param_name}: {param_value}")
                else:
                    normalized_params[key][param_name] = param_value
            
            # Unificar nombres de parámetros para kernels específicos
            self._unify_parameter_names(key, normalized_params[key])
        
        return normalized_params
    
    def _unify_parameter_names(self, dataset_key: str, params: Dict[str, Any]):
        """Unifica los nombres de parámetros inconsistentes"""
        kernel_type = dataset_key.split('_')[-1].lower()
        
        if kernel_type == 'poly':
            # Unificar parámetros gamma para poly
            if 'use_scale_gamma' in params:
                if params['use_scale_gamma']:
                    params['gamma'] = 'scale'
                else:
                    params['gamma'] = params.get('gamma_custom', 'scale')
                # Eliminar parámetros temporales
                params.pop('use_scale_gamma', None)
                params.pop('gamma_custom', None)
            
            # Asegurar que coef0 existe
            if 'coef0' not in params:
                params['coef0'] = 1.0
                
        elif kernel_type == 'rbf':
            # Unificar parámetros gamma para rbf
            if 'gamma_type' in params:
                if params['gamma_type'] == 'value':
                    params['gamma'] = params.get('gamma_value', 'scale')
                else:
                    params['gamma'] = params['gamma_type']
                # Eliminar parámetros temporales
                params.pop('gamma_type', None)
                params.pop('gamma_value', None)
        
        # Asegurar que C sea float
        if 'C' in params:
            try:
                params['C'] = float(params['C'])
            except (ValueError, TypeError):
                params['C'] = self.base_config['svm']['C']
    
    def build_kernel_with_params(self, kernel_type: str, params: Dict[str, Any]):
        """Construye el kernel manejando parámetros específicos de cada tipo"""
        try:
            # Preparar parámetros según el tipo de kernel
            kernel_params = {}
            
            if kernel_type == 'poly':
                kernel_params = {
                    'degree': params.get('degree', 3),
                    'gamma': params.get('gamma', 'scale'),
                    'coef0': params.get('coef0', 1.0)
                }
                
            elif kernel_type == 'rbf':
                kernel_params = {
                    'gamma': params.get('gamma', 'scale')
                }
                
            elif kernel_type == 'custom_hermite':
                kernel_params = {
                    'degree': params.get('degree', 3)
                }
                
            elif kernel_type == 'custom_gegen':
                kernel_params = {
                    'degree': params.get('degree', 3),
                    'alpha': params.get('alpha', 1.0)
                }
                
            elif kernel_type == 'custom_alsalam':
                kernel_params = {
                    'degree': params.get('degree', 3),
                    'a': params.get('a', -1),
                    'q': params.get('q', 0.5)
                }
                
            elif kernel_type == 'linear':
                kernel_params = {}  # Linear no necesita parámetros adicionales
            
            logger.debug(f"🔧 Parámetros para {kernel_type}: {kernel_params}")
            kernel_info = build_single_kernel(kernel_type, **kernel_params)
            
            if kernel_info:
                return kernel_info
            else:
                logger.error(f"❌ build_single_kernel retornó None para {kernel_type}")
                return None
                
        except Exception as e:
            logger.error(f"💥 Error construyendo kernel {kernel_type}: {e}")
            return None
    
    def evaluate_with_statistical_rigor(self, n_runs: int = 35, confidence_level: float = 0.95):
        """Ejecuta evaluación estadísticamente rigurosa"""
        optimized_params = self.load_optimized_parameters()
        
        if not optimized_params:
            logger.error("❌ No hay parámetros optimizados para evaluar")
            return None, None
        
        all_results = []
        detailed_results = []
        
        for dataset_name, X, y in self.datasets:
            logger.info(f"\n📊 Evaluando dataset: {dataset_name}")
            
            # Obtener todos los kernels optimizados para este dataset
            dataset_keys = [k for k in optimized_params.keys() if k.startswith(f"{dataset_name}_")]
            
            if not dataset_keys:
                logger.warning(f"⚠️ No se encontraron parámetros optimizados para {dataset_name}")
                continue
            
            for key in dataset_keys:
                kernel_type = key.split('_')[-1].lower()
                params = optimized_params[key]
                
                # Reconstruir el nombre completo del kernel
                full_kernel_type = kernel_type
                if kernel_type in ['hermite', 'gegen', 'alsalam']:
                    full_kernel_type = f"custom_{kernel_type}"
                
                logger.info(f"🔧 Evaluando {full_kernel_type} con {n_runs} ejecuciones")
                logger.debug(f"📋 Parámetros: {params}")
                
                # Construir kernel con parámetros optimizados usando el nuevo método
                kernel_info = self.build_kernel_with_params(full_kernel_type, params)
                if not kernel_info:
                    logger.error(f"❌ No se pudo construir kernel {full_kernel_type}")
                    continue
                
                kernel_name = kernel_info['name']
                kernel_func = kernel_info['func']
                C_value = params.get('C', self.base_config['svm']['C'])
                
                # Validar tipo de C
                try:
                    C_value = float(C_value)
                except (ValueError, TypeError):
                    logger.warning(f"⚠️ C inválido: {C_value}. Usando valor por defecto.")
                    C_value = self.base_config['svm']['C']
                
                # Ejecutar múltiples veces con diferentes semillas
                run_accuracies = []
                run_f1_scores = []
                run_times = []
                successful_runs = 0
                
                for run_idx in range(n_runs):
                    try:
                        # Usar semilla diferente para cada ejecución
                        random_seed = 42 + run_idx

                        # Medir el tiempo de entrenamiento
                        start_time = time.perf_counter()
                        try:
                            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                                future = executor.submit(
                                    train_svm,
                                    X, y,
                                    kernel_func,
                                    C_value,
                                    self.base_config['cross_validation']['n_splits'],
                                    1,
                                    X.shape[1],
                                    random_seed
                                )
                                metrics = future.result(timeout=TIMEOUT)
                        except concurrent.futures.TimeoutError:
                            logger.warning(f"   ⏰ Ejecución {run_idx} excedió el tiempo límite de {TIMEOUT}s y fue abortada")
                            continue
                        elapsed_time = time.perf_counter() - start_time

                        run_accuracies.append(metrics['accuracy'])
                        run_f1_scores.append(metrics['f1_score'])
                        run_times.append(elapsed_time)
                        successful_runs += 1

                        # Guardar resultado detallado
                        detailed_results.append({
                            'dataset': dataset_name,
                            'kernel': kernel_name,
                            'run': run_idx,
                            'accuracy': metrics['accuracy'],
                            'f1_score': metrics['f1_score'],
                            'train_time': elapsed_time,
                            'random_seed': random_seed
                        })
                        
                        if (run_idx + 1) % 10 == 0:
                            logger.info(f"   ✅ Completadas {run_idx + 1}/{n_runs} ejecuciones ({successful_runs} exitosas)")
                            
                    except Exception as e:
                        logger.warning(f"   ⚠️ Ejecución {run_idx} falló: {e}")
                        continue
                
                if run_accuracies:
                    # Calcular estadísticas
                    acc_mean = np.mean(run_accuracies) 
                    acc_std = np.std(run_accuracies) 
                    acc_ci = self.calculate_confidence_interval(run_accuracies, confidence_level)
                    acc_ci = (acc_ci[0] , acc_ci[1] )  # Convertir a porcentaje
                    
                    f1_mean = np.mean(run_f1_scores) 
                    f1_std = np.std(run_f1_scores) 
                    f1_ci = self.calculate_confidence_interval(run_f1_scores, confidence_level)
                    f1_ci = (f1_ci[0] , f1_ci[1] )
                    
                    time_mean = np.mean(run_times)
                    time_std = np.std(run_times)
                    
                    # Test de normalidad
                    normality_test = stats.shapiro(run_accuracies) if len(run_accuracies) >= 3 else (None, 1.0)
                    
                    result = {
                        'dataset': dataset_name,
                        'kernel': kernel_name,
                        'n_runs': len(run_accuracies),
                        'successful_runs': successful_runs,
                        'accuracy_mean': acc_mean,
                        'accuracy_std': acc_std,
                        'accuracy_ci_lower': acc_ci[0],
                        'accuracy_ci_upper': acc_ci[1],
                        'accuracy_min': np.min(run_accuracies) ,
                        'accuracy_max': np.max(run_accuracies) ,
                        'f1_mean': f1_mean,
                        'f1_std': f1_std,
                        'f1_ci_lower': f1_ci[0],
                        'f1_ci_upper': f1_ci[1],
                        'time_mean': time_mean,
                        'time_std': time_std,
                        'normality_pvalue': normality_test[1],
                        'best_params': params
                    }
                    
                    all_results.append(result)
                    
                    logger.info(f"📈 {dataset_name}-{kernel_name}: "
                               f"Accuracy = {acc_mean:.2f}% ± {acc_std:.2f}% "
                               f"(95% CI: {acc_ci[0]:.2f}% - {acc_ci[1]:.2f}%)")
                else:
                    logger.warning(f"⚠️ No se completaron ejecuciones exitosas para {dataset_name}-{kernel_name}")
        
        # Guardar resultados
        self.save_results(all_results, detailed_results)
        return all_results, detailed_results
    
    def calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calcula intervalo de confianza"""
        if len(data) < 2:
            return (np.mean(data), np.mean(data))
        
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)  # Error estándar de la media
        
        if n >= 30:  # Usar distribución normal para n grande
            ci = stats.norm.interval(confidence, loc=mean, scale=sem)
        else:  # Usar distribución t de Student para n pequeño
            ci = stats.t.interval(confidence, n-1, loc=mean, scale=sem)
        
        return ci
    
    def save_results(self, summary_results: List[Dict], detailed_results: List[Dict]):
        """Guarda resultados en archivos"""
        output_dir = Path("statistical_results")
        output_dir.mkdir(exist_ok=True)
        
        # Guardar resumen estadístico
        if summary_results:
            df_summary = pd.DataFrame(summary_results)
            summary_file = output_dir / "statistical_summary.csv"
            df_summary.to_csv(summary_file, index=False)
            logger.info(f"💾 Resumen estadístico guardado en {summary_file}")
        
        # Guardar resultados detallados
        if detailed_results:
            df_detailed = pd.DataFrame(detailed_results)
            detailed_file = output_dir / "detailed_results.csv"
            df_detailed.to_csv(detailed_file, index=False)
            logger.info(f"💾 Resultados detallados guardados en {detailed_file}")
        
        # Generar reporte estadístico
        if summary_results:
            self.generate_statistical_report(pd.DataFrame(summary_results), output_dir)
    
    def generate_statistical_report(self, df: pd.DataFrame, output_dir: Path):
        """Genera un reporte estadístico completo"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("📊 REPORTE ESTADÍSTICO - EVALUACIÓN RIGUROSA")
        report_lines.append("=" * 80)
        report_lines.append(f"Total de combinaciones evaluadas: {len(df)}")
        report_lines.append(f"Número de ejecuciones por combinación: {df['n_runs'].iloc[0] if len(df) > 0 else 0}")
        report_lines.append("")
        
        # Por dataset
        for dataset in df['dataset'].unique():
            report_lines.append(f"\n🏆 DATASET: {dataset}")
            report_lines.append("-" * 50)
            
            df_ds = df[df['dataset'] == dataset].sort_values('accuracy_mean', ascending=False)
            
            for _, row in df_ds.iterrows():
                report_lines.append(
                    f"{row['kernel']:12} | "
                    f"Accuracy: {row['accuracy_mean']:.2f}% ± {row['accuracy_std']:.2f}% | "
                    f"CI95%: [{row['accuracy_ci_lower']:.2f}%, {row['accuracy_ci_upper']:.2f}%] | "
                    f"F1: {row['f1_mean']:.2f}%"
                )
        
        # Mejores resultados globales
        report_lines.append("\n" + "=" * 80)
        report_lines.append("🥇 MEJORES RESULTADOS GLOBALES")
        report_lines.append("=" * 80)
        
        if not df.empty:
            best_overall = df.loc[df['accuracy_mean'].idxmax()]
            report_lines.append(f"Mejor combinación global: {best_overall['dataset']} - {best_overall['kernel']}")
            report_lines.append(f"Accuracy: {best_overall['accuracy_mean']:.2f}% ± {best_overall['accuracy_std']:.2f}%")
            report_lines.append(f"Intervalo de confianza 95%: [{best_overall['accuracy_ci_lower']:.2f}%, {best_overall['accuracy_ci_upper']:.2f}%]")
        else:
            report_lines.append("❌ No hay resultados para mostrar")
        
        # Guardar reporte
        report_file = output_dir / "statistical_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"📄 Reporte estadístico guardado en {report_file}")
        
        # Imprimir resumen en consola
        print('\n'.join(report_lines[-10:]))

def main():
    """Función principal"""
    evaluator = StatisticalEvaluator()
    
    logger.info("🚀 INICIANDO EVALUACIÓN ESTADÍSTICAMENTE RIGUROSA")
    logger.info("📚 Ejecutando 35 veces cada combinación dataset-kernel")
    logger.info("⏰ Esto tomará tiempo...")
    
    try:
        summary_results, detailed_results = evaluator.evaluate_with_statistical_rigor(n_runs=35)
        
        if summary_results:
            logger.info("✅ Evaluación estadística completada!")
            logger.info("📊 Resultados guardados en directorio 'statistical_results/'")
        else:
            logger.error("❌ La evaluación no produjo resultados")
            
    except KeyboardInterrupt:
        logger.info("⏹️ Evaluación interrumpida por el usuario")
    except Exception as e:
        logger.error(f"💥 Error durante la evaluación: {e}")

if __name__ == "__main__":
    main()