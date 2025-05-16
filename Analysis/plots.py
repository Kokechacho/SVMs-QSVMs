# analysis/plots.py
import pandas as pd
import matplotlib.pyplot as plt


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