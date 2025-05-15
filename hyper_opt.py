import itertools
import json
import os
import subprocess
import shutil
import pandas as pd
from pathlib import Path

# Configuración base
BASE_CONFIG = "config.json"
OUTPUT_BASE_DIR = "hparam_results"
CSV_NAME = "svm_results.csv"

# Espacio de búsqueda de hiperparámetros (personalizable)
SEARCH_SPACE = {
    "svm.C": [10],                      # Valores de C
    "svm.poly.degree": [2],                   # Grados para kernel polinomial
    "svm.custom_hermite.degree": [3],         # Grado para Hermite
    "svm.custom_gegen.alpha": [0.1]        # Alpha para Gegenbauer
}

def generate_combinations(search_space):
    """
    Para cada C en search_space['svm.C'], produce una combinación
    tomando en paralelo (por índice) los valores de los demás parámetros.
    """
    # 1) Extraemos los valores de C
    C_values = search_space["svm.C"]
    # 2) Preparamos el resto de parámetros (excluimos 'svm.C')
    other_keys   = [k for k in search_space if k != "svm.C"]
    other_values = [search_space[k] for k in other_keys]
    
    # 3) Zip para iterar por índice (se detiene al más corto)
    zipped = list(zip(*other_values))
    
    # 4) Para cada C y cada tupla paralela de valores:
    for C in C_values:
        for vals in zipped:
            combo = {"svm.C": C}
            combo.update({ key: val for key, val in zip(other_keys, vals) })
            yield combo

def update_config(base_config, params):
    """Actualiza la configuración con los nuevos parámetros."""
    cfg = base_config.copy()
    for param_path, value in params.items():
        parts = param_path.split('.')
        current = cfg
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return cfg

def main():
    # Cargar configuración base
    with open(BASE_CONFIG, 'r') as f:
        base_cfg = json.load(f)
    
    # Crear directorio base para resultados
    Path(OUTPUT_BASE_DIR).mkdir(exist_ok=True)
    
    all_results = []
    for i, params in enumerate(generate_combinations(SEARCH_SPACE)):
        print(f"\n=== Combinación {i+1} ===")
        print("Parámetros:", params)
        
        # Crear directorio para esta ejecución
        run_dir = os.path.join(OUTPUT_BASE_DIR, f"run_{i+1}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Actualizar configuración
        cfg = update_config(base_cfg, params)
        cfg["output"]["results_dir"] = os.path.join(run_dir, "results")
        cfg["output"]["plots_dir"] = os.path.join(run_dir, "plots")
        
        # Guardar configuración temporal
        temp_config_path = os.path.join(run_dir, "config_temp.json")
        with open(temp_config_path, 'w') as f:
            json.dump(cfg, f, indent=4)
        
        # Ejecutar experiments.py
        try:
            subprocess.run(
                ["python", "experiments.py", "--config", temp_config_path, "--no-plots"],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error en ejecución {i+1}: {e.stderr}")
            continue
        
        # Recopilar resultados
        results_path = os.path.join(cfg["output"]["results_dir"], CSV_NAME)
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            df["run_id"] = i+1
            all_results.append(df)
        else:
            print(f"Resultados no encontrados en {results_path}")
    
    # Consolidar todos los resultados
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_csv = os.path.join(OUTPUT_BASE_DIR, "all_results.csv")
        final_df.to_csv(final_csv, index=False)
        print(f"\nResultados consolidados guardados en {final_csv}")
        
        # --- Nuevo análisis por (dataset, kernel) ---
        # Agrupar por dataset y kernel, y encontrar el máximo accuracy en cada grupo
        best_indices = final_df.groupby(['dataset', 'kernel'])['accuracy'].idxmax()
        best_results = final_df.loc[best_indices]
        
        # Formatear la tabla
        print("\n=== Mejores parámetros por (Dataset, Kernel) ===")
        print("="*90)
        print(f"{'Dataset':<15} | {'Kernel':<15} | {'Accuracy':<8} | {'C':<5} | {'Degree':<6} | {'Gamma':<6} | {'Alpha':<6} |")
        print("-"*90)
        
        for _, row in best_results.iterrows():
            # Extraer parámetros específicos del kernel
            kernel = row['kernel']
            params = {
                'C': row.get('svm.C', '-'),
                'degree': row.get('svm.poly.degree', '-'),
                'gamma': row.get('svm.poly.gamma', row.get('svm.rbf.gamma', '-')),
                'alpha': row.get('svm.custom_gegen.alpha', '-')
            }
            
            # Imprimir fila formateada
            print(f"{row['dataset']:<15} | {kernel:<15} | {row['accuracy']:.4f}    | "
                f"{params['C']:<5} | {params['degree']:<6} | {params['gamma']:<6} | {params['alpha']:<6} |")
        
        print("="*90)

    else:
        print("No se encontraron resultados válidos.")

if __name__ == "__main__":
    main()