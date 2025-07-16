# analysis/plots.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_metrics_comparison(results, metrics=['accuracy', 'f1_score', 'n_support_vectors'], save_path=None, show: bool = True):
    """
    Plot bar charts comparing metrics per kernel.
    :param results: List of dicts per kernel for a given dataset
    :param metrics: Which metrics to plot
    :param save_path: If provided, path to save the figure as .png
    """
    import matplotlib.pyplot as plt
    results = [r for r in results if r.get('pca_dim') is None]
    if not results:
        return  # no hay nada que bar‑plotear

    kernel_names = [r['kernel'] for r in results]

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]  # para poder iterar

    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        bars = axes[i].bar(kernel_names, values, color='#1f77b4')
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width() / 2, height / 2,
                f'{height:.2f}', ha='center', va='center', color='white', fontweight='bold'
            )

    plt.suptitle(f"Métricas por kernel ({results[0]['dataset']})", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[Plot] Guardado en {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def plot_accuracy_diff_table(
    df_comp: pd.DataFrame,
    show: bool = True
) -> None:
    """
    Muestra una tabla con la diferencia de accuracy entre observado y esperado.

    :param df_comp: DataFrame con columnas ['dataset','kernel','acc_diff']
    """
    # 1) Validar entrada
    if df_comp is None or df_comp.empty:
        print("No hay datos para mostrar en la tabla de diferencias de accuracy.")
        return

    # 2) Pivot para formatear la tabla
    try:
        pivot = df_comp.pivot(index='dataset', columns='kernel', values='acc_diff')
    except Exception as e:
        print(f"Error al pivotar df_comp: {e}")
        return

    if pivot.empty:
        print("El pivot de diferencias de accuracy está vacío.")
        return

    # 3) Dibujar tabla
    fig, ax = plt.subplots(
        figsize=(len(pivot.columns) * 1.2, len(pivot.index) * 0.5 + 1)
    )
    ax.axis('off')
    tbl = ax.table(
        cellText=pivot.values.round(2),
        rowLabels=pivot.index,
        colLabels=pivot.columns,
        cellLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    plt.title('Diferencia Accuracy Observado vs Esperado')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()

def plot_pca_evolution(
    results: list[dict],
    metrics: list[str],
    save_path: str | None = None,
    show: bool = True
):
    """
    Dibuja la evolución de múltiples métricas (accuracy, f1_score, etc.)
    en función de la dimensión PCA (`pca_dim`), con una línea por kernel.
    
    :param results: lista de registros con campos 'kernel', 'pca_dim' y métricas.
    :param metrics: lista de métricas a graficar.
    :param save_path: ruta base para guardar las imágenes (sin la métrica).
    :param show: si True, muestra las gráficas.
    """
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No hay datos PCA disponibles.")
        return
    if 'pca_dim' not in df.columns:
        # No hay datos de PCA, salir sin error
        return
    
    df = df.dropna(subset=['pca_dim'])
    kernels = df['kernel'].unique()

    for metric in metrics:
        if metric not in df.columns:
            print(f"[Advertencia] Métrica `{metric}` no encontrada.")
            continue

        fig, ax = plt.subplots(figsize=(8,5))
        for k in kernels:
            sub = df[df['kernel'] == k].sort_values('pca_dim')
            ax.plot(sub['pca_dim'], sub[metric], marker='o', label=k)

        ax.set_xlabel('n_components (PCA)')
        ax.set_ylabel(metric.replace('_',' ').title())
        dataset_name = df["dataset"].unique()[0]
        ax.set_title(f'Evolución de {metric.replace("_", " ").title()} vs. PCA dim - {dataset_name}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        if save_path:
            metric_filename = f"{save_path.rstrip('.png')}_{metric}.png"
            plt.savefig(metric_filename)
            print(f"[Plot PCA] Guardado en {metric_filename}")

        if show:
            plt.show()
        else:
            plt.close()

def plot_pca_evolution_global(
    records: list,
    metrics: list,
    save_path: str,
    show: bool = False
):
    """
    Plotea para cada métrica la evolución global de PCA combinando TODOS los datasets:
    - media (línea gruesa y opaca)
    - mediana (línea fina y semitransparente)
    - banda std (fill_between, muy transparente)

    Limpia antes los valores no numéricos o infinitos de 'pca_dim'.
    """

    # DataFrame inicial
    df = pd.DataFrame(records)

    if df.empty:
        print("No hay datos PCA disponibles.")
        return
    if 'pca_dim' not in df.columns:
        # No hay datos de PCA, salir sin error
        return

    # Forzar numérico y eliminar NA/inf
    df['pca_dim'] = pd.to_numeric(df['pca_dim'], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['pca_dim'])

    # Ahora sí, casteo seguro a entero y ordenamos
    df['pca_dim'] = df['pca_dim'].astype(int)
    df = df.sort_values('pca_dim')

    # Agrupar por dimensión de PCA
    grp = df.groupby('pca_dim')

    # Asegurarnos de que exista la carpeta
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Dibujar cada métrica
    for metric in metrics:
        # Estadísticos
        mean_   = grp[metric].mean()
        median_ = grp[metric].median()
        std_    = grp[metric].std()

        dims = mean_.index.values

        plt.figure(figsize=(8, 5))
        # Banda de desviación típica
        plt.fill_between(
            dims,
            mean_ - std_,
            mean_ + std_,
            alpha=0.2,
            label=f'{metric} ± std'
        )
        # Mediana
        plt.plot(
            dims,
            median_,
            linestyle='--',
            linewidth=1.5,
            alpha=0.6,
            label=f'median {metric}'
        )
        # Media
        plt.plot(
            dims,
            mean_,
            linestyle='-',
            linewidth=2.5,
            alpha=1.0,
            label=f'mean {metric}'
        )

        plt.xlabel('Número de componentes PCA')
        plt.ylabel(metric.replace('_', ' ').capitalize())
        plt.title(f'Evolución global PCA ({metric})')
        plt.legend()
        plt.grid(alpha=0.3)

        out_file = f"{save_path}_{metric}.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        if show:
            plt.show()
        plt.close()


def plot_pca_evolution_all_kernels(
    records: list,
    metrics: list,
    save_path: str,
    show: bool = False,
    plot_median: bool = False,
    plot_std: bool = False
):
    """
    Para cada kernel en `records`, genera la gráfica de evolución PCA:
      - cada dataset en alpha baja
      - media destacada
      - opcional: mediana y banda ±std

    Se crean archivos:
      {save_path}_{kernel}_{metric}.png

    Parameters
    ----------
    records : list of dict
    metrics : list of str
    save_path : str
      Ruta base (sin extensión ni kernel), p. ej. ".../plots/kernel_pca"
    show : bool
    plot_median : bool
    plot_std : bool
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    df = pd.DataFrame(records)

    if df.empty:
        print("No hay datos PCA disponibles.")
        return
    if 'pca_dim' not in df.columns:
        # No hay datos de PCA, salir sin error
        return
    
    # Limpiar pca_dim igual que antes
    df['pca_dim'] = pd.to_numeric(df['pca_dim'], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['pca_dim'])
    df['pca_dim'] = df['pca_dim'].astype(int)

    kernels = df['kernel'].unique()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for kernel in kernels:
        df_k = df[df['kernel'] == kernel]
        if df_k.empty:
            continue

        for metric in metrics:
            plt.figure(figsize=(8, 5))

            # líneas de cada dataset
            for ds_name, grp_ds in df_k.groupby('dataset'):
                plt.plot(
                    grp_ds['pca_dim'],
                    grp_ds[metric],
                    linewidth=1.0,
                    alpha=0.2,
                    label=ds_name if metric == metrics[0] else None
                )

            # estadísticos
            grp = df_k.groupby('pca_dim')[metric]
            mean_ = grp.mean()
            if plot_std:
                std_ = grp.std()
                plt.fill_between(
                    mean_.index,
                    mean_ - std_,
                    mean_ + std_,
                    alpha=0.2,
                    label='±1 std'
                )
            if plot_median:
                median_ = grp.median()
                plt.plot(
                    median_.index,
                    median_.values,
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.6,
                    label='median'
                )

            # media destacada
            plt.plot(
                mean_.index,
                mean_.values,
                linestyle='-',
                linewidth=3.0,
                alpha=1.0,
                label='mean'
            )

            plt.xlabel('Número de componentes PCA')
            plt.ylabel(metric.replace('_', ' ').capitalize())
            plt.title(f'Kernel={kernel} · Evolución PCA ({metric})')
            plt.legend(loc='best', fontsize='small', ncol=2)
            plt.grid(alpha=0.3)

            out_file = f"{save_path}_{kernel}_{metric}.png"
            plt.tight_layout()
            plt.savefig(out_file, dpi=300)
            if show:
                plt.show()
            plt.close()

def plot_train_vs_val_metrics(
    records: list,
    metrics: list,
    save_path: str,
    show: bool = False
):
    """
    Genera un único gráfico con subplots, uno por cada métrica,
    comparando para cada kernel:
      - valor medio en validación ± std
      - valor medio en entrenamiento ± std

    Se usa un diagrama de barras con barras adyacentes y barras de error.

    Parameters
    ----------
    records : list of dict
        Cada dict debe tener:
          'kernel',
          para cada m en metrics: m, m + '_std',
          para cada m en metrics: 'train_' + m, 'train_' + m + '_std'
    metrics : list of str
        Métricas a plotear (e.g. ['accuracy', 'f1_score']).
    save_path : str
        Ruta de salida del PNG.
    show : bool
        Si True, muestra el plot.
    """
    df = pd.DataFrame(records)
    kernels = sorted(df['kernel'].unique())
    n_metrics = len(metrics)

    # Configurar figura con un subplot por métrica
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(5 * n_metrics, 6),
        squeeze=False
    )

    for idx, metric in enumerate(metrics):
        ax = axes[0, idx]
        val_col, val_std = metric, metric + '_std'
        tr_col, tr_std = 'train_' + metric, 'train_' + metric + '_std'

        grp = df.groupby('kernel')
        mean_val = grp[val_col].mean().reindex(kernels).values
        err_val = grp[val_std].mean().reindex(kernels).values
        mean_tr = grp[tr_col].mean().reindex(kernels).values
        err_tr = grp[tr_std].mean().reindex(kernels).values

        x = np.arange(len(kernels))
        width = 0.35

        # Barras de validación y entrenamiento
        ax.bar(
            x - width/2,
            mean_val,
            width,
            yerr=err_val,
            capsize=3,
            label='Validation'
        )
        ax.bar(
            x + width/2,
            mean_tr,
            width,
            yerr=err_tr,
            capsize=3,
            label='Training'
        )

        ax.set_xticks(x)
        ax.set_xticklabels(kernels, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').capitalize())
        ax.set_title(metric.replace('_', ' ').capitalize())
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
