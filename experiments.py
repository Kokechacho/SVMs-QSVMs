import logging
import os
import json
import pandas as pd
import time
from pathlib import Path
import argparse

from Data.loader import load_all_datasets, load_expected_results
from Kernels.all_kernels import build_kernels
from Model.train import train_svm
from Analysis.plots import (
    plot_metrics_comparison,
    plot_accuracy_diff_table,
    plot_decision_boundary
)

from sklearn.model_selection import train_test_split

def main(config_path: str, do_plots: bool = True):
    # --- 1. Leer configuración ---
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    # Logging
    os.makedirs(cfg['output']['results_dir'], exist_ok=True)
    os.makedirs(cfg['output']['plots_dir'], exist_ok=True)
    logging.basicConfig(level=getattr(logging, cfg['output']['log_level']))
    log = logging.getLogger(__name__)

    # --- 2. Cargar datos ---
    data_cfg = cfg['data']
    datasets = load_all_datasets(
        uci_list      = data_cfg['uci'],
        libsvm_dir    = Path(data_cfg['libsvm_dir']),
        libsvm_files  = data_cfg['libsvm']
    )
    df_expected = load_expected_results()

    # --- 3. Construir lista de kernels ---
    svm_cfg = cfg['svm']
    kernels = build_kernels(
        hermite_degree=svm_cfg['custom_hermite']['degree'],
        gegen_degree=svm_cfg['custom_gegen']['degree'],
        gegen_alpha=svm_cfg['custom_gegen']['alpha']
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
                C=svm_cfg['C'],
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
                'svm.C': svm_cfg['C'],  
                'svm.poly.degree': svm_cfg['poly']['degree'] if name == "POLY" else None,
                'svm.poly.gamma': svm_cfg['poly']['gamma'] if name == "POLY" else None,
                'svm.custom_hermite.degree': svm_cfg['custom_hermite']['degree'] if name == "HERMITE" else None,
                'svm.custom_gegen.alpha': svm_cfg['custom_gegen']['alpha'] if name == "GEGEN" else None,
                **metrics
            }
            records.append(rec)

        # --- 5. Plot métricas y tabla de diferencias ---
        df_ds = pd.DataFrame([r for r in records if r['dataset'] == ds_name])
        # Normalizar nombres de datasets
        df_ds['dataset'] = df_ds['dataset'].str.strip().str.lower()
        df_expected['dataset'] = df_expected['dataset'].str.strip().str.lower()

        # Merge observado vs esperado
        df_merge = pd.merge(
            df_ds[['dataset', 'kernel', 'accuracy']],
            df_expected,
            on=['dataset', 'kernel'],
            suffixes=('_obs', '_exp')
        )
        df_merge = df_merge.assign(
            acc_diff=lambda df: df['accuracy_obs'] - df['accuracy_exp']
        )

        # Ejecutar gráficos solo si do_plots=True
        plot_accuracy_diff_table(
            df_merge[['dataset', 'kernel', 'acc_diff']],
            show=do_plots
        )

        # Métricas comparativas (siempre guardadas; mostrar según do_plots)
        plot_path = os.path.join(cfg['output']['plots_dir'], f"{ds_name}_metrics.png")
        plot_metrics_comparison(
            df_ds.to_dict(orient='records'),
            metrics=['accuracy', 'f1_score', 'support_vector_acc', 'train_time'],
            save_path=plot_path,
            show=do_plots
        )

    # --- 6. Guardar resultados globales ---
    df_results = pd.DataFrame(records)
    out_csv = os.path.join(cfg['output']['results_dir'], "svm_results.csv")
    df_results.to_csv(out_csv, index=False)
    log.info(f"Resultados guardados en {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ejecuta experimentos de SVM con configuración JSON y control de plots."
    )
    parser.add_argument(
        "--config", "-c",
        dest="config_path",
        default=str(Path(__file__).parent / "config.json"),
        help="Ruta al archivo de configuración JSON (por defecto: config.json en la raíz del proyecto)."
    )
    parser.add_argument(
        "--no-plots",
        dest="do_plots",
        action="store_false",
        help="Desactiva la ejecución de las secciones de plotting (por defecto activado)."
    )
    parser.set_defaults(do_plots=True)
    args = parser.parse_args()
    main(args.config_path, args.do_plots)
