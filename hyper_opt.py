import json
import pandas as pd
import numpy as np
import optuna
from pathlib import Path
import logging
from typing import Dict, Any, List
from Data.loader import load_all_datasets
from Kernels.all_kernels import build_single_kernel
from Model.train import train_svm

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BayesianOptimizer:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.load_base_config()
        self.load_datasets()
        
    def load_base_config(self):
        """Carga la configuración base"""
        with open(self.config_path, 'r') as f:
            self.base_config = json.load(f)
    
    def load_datasets(self):
        """Carga todos los datasets una sola vez"""
        data_cfg = self.base_config['data']
        self.datasets = load_all_datasets(
            uci_list=data_cfg['uci'],
            libsvm_dir=Path(data_cfg['libsvm_dir']),
            libsvm_files=data_cfg['libsvm']
        )
        
        # Filtrar datasets pequeños para prueba rápida (opcional)
        self.datasets = [ds for ds in self.datasets if ds[1].shape[0] < 10000]  # Solo datasets con < 10k muestras
        logger.info(f"📊 Cargados {len(self.datasets)} datasets")
    
    def get_kernel_function(self, kernel_name: str, params: Dict[str, Any]):
        """Obtiene la función del kernel con parámetros específicos"""
        kernel_info = build_single_kernel(kernel_name, **params)
        return kernel_info['func'] if kernel_info else None
    
    def objective(self, trial, dataset_name: str, X: np.ndarray, y: np.ndarray, kernel_type: str):
        """Función objetivo para Optuna"""
        params = {'C': trial.suggest_int('C', 1, 100)}
        
        # Espacios de búsqueda específicos por kernel
        if kernel_type == 'poly':
            params['degree'] = trial.suggest_int('degree', 1, 6)
            
            # Versión simplificada para debug
            params['coef0'] = trial.suggest_float('coef0', 2**(-6), 2**2)
                
        elif kernel_type == 'rbf':
            gamma_type = trial.suggest_categorical('gamma_type', ['scale', 'auto', 'value'])
            if gamma_type == 'value':
                params['gamma'] = trial.suggest_float('gamma_value', 2**(-6), 2**2)
            else:
                params['gamma'] = gamma_type
                
        elif kernel_type == 'custom_hermite':
            params['degree'] = trial.suggest_int('degree', 1, 6)
            
        elif kernel_type == 'custom_gegen':
            params['degree'] = trial.suggest_int('degree', 1, 6)
            params['alpha'] = trial.suggest_float('alpha', -0.49, 1.5)
            
        elif kernel_type == 'custom_alsalam':
            params['degree'] = trial.suggest_int('degree', 1, 7)
            params['a'] = trial.suggest_categorical('a', [-1])
            params['q'] = trial.suggest_float('q', 0.01, 0.99, step=0.01)
        
        try:
            # Obtener función del kernel
            kernel_func = self.get_kernel_function(kernel_type, params)
            if kernel_func is None:
                logger.warning(f"No se pudo construir kernel {kernel_type} con params {params}")
                return 0.0
            
            # Entrenar modelo - SIN PCA para optimización
            n_feats = X.shape[1]
            metrics = train_svm(
                X, y,
                kernel_func=kernel_func,
                C=params['C'],
                cv=min(10, self.base_config['cross_validation']['n_splits']),  # Menos folds para velocidad
                n_jobs=1,  # Un solo job para evitar conflictos
                pca_dim=None  # Sin PCA para optimización
            )
            
            accuracy = metrics['accuracy']
            logger.info(f"✅ {dataset_name}-{kernel_type} - Trial {trial.number}: {accuracy:.4f}")
            
            return accuracy
            
        except Exception as e:
            logger.warning(f"⚠️ Trial {trial.number} falló: {e}")
            return 0.0
    
    def optimize_all(self, n_trials: int = 35):
        """Ejecuta optimización para combinaciones dataset-kernel"""
        results = []
        
        # Definir kernels a optimizar
        kernels_to_optimize = ['linear', 'poly', 'rbf', 'custom_hermite', 'custom_gegen', 'custom_alsalam']
        
        for dataset_name, X, y in self.datasets: 
            for kernel_type in kernels_to_optimize:  
                logger.info(f"\n🎯 Optimizando {dataset_name} - {kernel_type}")
                if kernel_type == 'custom_alsalam':
                    try:
                        study = optuna.create_study(
                            direction='maximize',
                            study_name=f"{dataset_name}_{kernel_type}"
                        )
                        
                        # Función objetivo parcial
                        def objective_partial(trial):
                            return self.objective(trial, dataset_name, X, y, kernel_type)
                        
                        study.optimize(objective_partial, n_trials=n_trials, show_progress_bar=True)
                        
                        if study.best_trial and study.best_value > 0:
                            # Convertir nombre del kernel para consistencia
                            kernel_display_name = kernel_type.upper().replace('CUSTOM_', '')
                            
                            result = {
                                'dataset': dataset_name,
                                'kernel': kernel_display_name,
                                'best_accuracy': study.best_value,
                                'best_params': study.best_params,
                                'n_trials': len(study.trials),
                                'best_trial_number': study.best_trial.number
                            }
                            results.append(result)
                            
                            logger.info(f"🏆 Mejor accuracy: {study.best_value:.4f}")
                            logger.info(f"⚙️ Mejores parámetros: {study.best_params}")
                        else:
                            logger.warning(f"❌ No se encontraron trials válidos para {dataset_name}-{kernel_type}")
                            
                    except Exception as e:
                        logger.error(f"💥 Error en {dataset_name}-{kernel_type}: {e}")
                        continue
            
            # Guardar resultados
            self.save_results(results)
        return results
    
    def save_results(self, results: List[Dict]):
        """Guarda los resultados de optimización sin sobrescribir los mejores anteriores."""
        output_dir = Path("optimization_results")
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / "bayesian_optimization_results.csv"
        config_file = output_dir / "best_hyperparameters.json"

        # 1️⃣ Cargar resultados previos (si existen)
        if results_file.exists():
            old_df = pd.read_csv(results_file)
        else:
            old_df = pd.DataFrame(columns=['dataset', 'kernel', 'best_accuracy', 'best_params', 'n_trials', 'best_trial_number'])
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                old_best_configs = json.load(f)
        else:
            old_best_configs = {}

        # 2️⃣ Combinar resultados nuevos con los antiguos, conservando los mejores
        combined_results = []

        for result in results:
            key = f"{result['dataset']}_{result['kernel']}"
            new_acc = result['best_accuracy']

            # Buscar si ya había un resultado anterior
            old_row = old_df[(old_df['dataset'] == result['dataset']) & (old_df['kernel'] == result['kernel'])]
            old_acc = old_row['best_accuracy'].values[0] if not old_row.empty else -np.inf

            if new_acc > old_acc:
                # Nuevo mejor resultado
                combined_results.append(result)
                old_best_configs[key] = result['best_params']
                logger.info(f"⬆️ Nuevo mejor resultado para {key}: {new_acc:.4f} (antes {old_acc:.4f})")
            else:
                # Mantener el anterior
                if not old_row.empty:
                    combined_results.append(old_row.iloc[0].to_dict())
                logger.info(f"➡️ Se mantiene el mejor resultado previo para {key}: {old_acc:.4f}")

        # 3️⃣ Guardar los resultados combinados
        df_results = pd.DataFrame(combined_results)
        df_results.to_csv(results_file, index=False)

        with open(config_file, 'w') as f:
            json.dump(old_best_configs, f, indent=2)

        logger.info(f"💾 Resultados actualizados en {output_dir}/")
        self.print_summary(combined_results)

    
    def print_summary(self, results: List[Dict]):
        """Imprime resumen de resultados"""
        print("\n" + "="*70)
        print("🎊 RESUMEN DE OPTIMIZACIÓN BAYESIANA")
        print("="*70)
        
        df = pd.DataFrame(results)
        if not df.empty:
            # Mejores resultados por dataset
            best_by_dataset = df.loc[df.groupby('dataset')['best_accuracy'].idxmax()]
            
            print("\n🏆 MEJORES RESULTADOS POR DATASET:")
            print("-" * 50)
            for _, row in best_by_dataset.iterrows():
                print(f"{row['dataset']:20} {row['kernel']:15} {row['best_accuracy']:.4f}")
            
            print(f"\n📈 Accuracy promedio: {df['best_accuracy'].mean():.4f} ± {df['best_accuracy'].std():.4f}")

def main():
    """Función principal"""
    optimizer = BayesianOptimizer()
    
    logger.info("🚀 Iniciando Optimización Bayesiana de Hiperparámetros")
    logger.info("⏰ Esto puede tomar varios minutos...")
    
    # Empezar con pocos trials para prueba
    results = optimizer.optimize_all(n_trials=100)
    
    if results:
        logger.info("✅ Optimización completada!")
    else:
        logger.error("❌ La optimización no produjo resultados")

if __name__ == "__main__":
    main()