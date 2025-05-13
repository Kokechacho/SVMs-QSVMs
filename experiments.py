# experiments.py
import logging
import os
import json
import pandas as pd
import time

from Data.loader import load_all_datasets, load_expected_results
from Kernels.all_kernels import build_kernels
from Model.train import train_svm, tune_svm
from Model.evaluate import evaluate_model
from Analysis.plots import (
    plot_metrics_comparison,
    plot_accuracy_diff_table,
    plot_decision_boundary
)

from sklearn.model_selection import train_test_split

def main(config_path: str = "config.json"):
    # --- 1. Leer configuración ---
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    # Logging
    os.makedirs(cfg['output']['results_dir'], exist_ok=True)
    os.makedirs(cfg['output']['plots_dir'], exist_ok=True)
    logging.basicConfig(level=getattr(logging, cfg['output']['log_level']))
    log = logging.getLogger(__name__)

    # --- 2. Cargar datos ---
    datasets = load_all_datasets()
    df_expected = load_expected_results()

    # --- 3. Construir lista de kernels ---
    svm_cfg = cfg['svm']
    kernels = build_kernels(
        hermite_degree=max(cfg['svm']['custom_hermite']['degree']),
        gegen_degree=max(cfg['svm']['custom_gegen']['degree']),
        gegen_alpha=cfg['svm']['custom_gegen']['alpha'][0]  # usa el primero como valor por defecto
    )

    # --- 4. Iterar datasets × kernels ---
    records = []
    for ds_name, X, y in datasets:
        log.info(f"=== Dataset: {ds_name} ===")

        # Para cada kernel
        for kern in kernels:
            name = kern['name']
            func = kern['func']
            log.info(f"Entrenando kernel: {name}")

            # Entrenar
            start = time.perf_counter()
            metrics = train_svm(
                X, y,
                kernel_func=func,
                C=svm_cfg['C'][0],
                cv=cfg['cross_validation']['n_splits'],
                n_jobs=cfg['cross_validation']['n_jobs']
            )
            elapsed = time.perf_counter() - start
            metrics['train_time'] = elapsed

            log.info(f"-> {name}: {metrics}")

            # Registrar
            rec = {
                'dataset': ds_name,
                'kernel': name,
                **metrics
            }
            records.append(rec)

        # --- 5. Plot métricas y tabla de diferencias ---
        df_ds = pd.DataFrame([r for r in records if r['dataset']==ds_name])
        # Normalizar
        df_ds['dataset']       = df_ds['dataset'].str.strip().str.lower()
        df_expected['dataset'] = df_expected['dataset'].str.strip().str.lower()

        # Hacemos el merge con sufijos para distinguir observada y esperada
        df_merge = pd.merge(
            df_ds[['dataset', 'kernel', 'accuracy']],
            df_expected,  # tiene columna 'accuracy'
            on=['dataset', 'kernel'],
            suffixes=('_obs', '_exp')
        )

        # Ahora sí hay columnas 'accuracy_obs' y 'accuracy_exp'
        df_merge = df_merge.assign(
            acc_diff=lambda df: df['accuracy_obs'] - df['accuracy_exp']
        )

        # Llamada a la tabla
        plot_accuracy_diff_table(df_merge[['dataset', 'kernel', 'acc_diff']])

        plot_path = os.path.join(cfg['output']['plots_dir'], f"{ds_name}_metrics.png")
        plot_metrics_comparison(
            df_ds.to_dict(orient='records'),
            metrics=['accuracy', 'f1_score', 'support_vector_acc', 'train_time'],
            save_path=plot_path
        )

    # --- 6. Guardar resultados globales ---
    df_results = pd.DataFrame(records)
    out_csv = os.path.join(cfg['output']['results_dir'], "svm_results.csv")
    df_results.to_csv(out_csv, index=False)
    log.info(f"Resultados guardados en {out_csv}")

if __name__ == "__main__":
    main()
