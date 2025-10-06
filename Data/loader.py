import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_svmlight_file

# Configuración de logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Rutas base
BASE_DIR = Path(__file__).parent.parent
LIBSVM_DIR = BASE_DIR / "Data" / "libsvm"

# Datos esperados de precisión por dataset y kernel
_EXPECTED_RESULTS = [
    {"dataset": "a1a",        "RBF": 83.24, "LINEAR": 83.55, "POLY": 83.50, "GCK": 83.48, "LEGEN": 80.26, "CHEB": 75.46, "HERMITE": 83.50, "GEGEN": 83.64},
    {"dataset": "australian", "RBF": 86.06, "LINEAR": 85.49, "POLY": 86.10, "GCK": 86.00, "LEGEN": 80.21, "CHEB": 79.95, "HERMITE": 86.15, "GEGEN": 86.51},
    {"dataset": "breast",     "RBF": 97.13, "LINEAR": 96.96, "POLY": 97.15, "GCK": 97.09, "LEGEN": 96.26, "CHEB": 95.37, "HERMITE": 97.37, "GEGEN": 97.29},
    {"dataset": "diabetes",   "RBF": 78.04, "LINEAR": 77.45, "POLY": 77.60, "GCK": 77.62, "LEGEN": 77.26, "CHEB": 70.23, "HERMITE": 77.94, "GEGEN": 77.52},
    {"dataset": "fourclass",  "RBF": 100.00,"LINEAR": 77.09, "POLY": 81.47, "GCK": 99.24, "LEGEN": 99.70, "CHEB": 99.95, "HERMITE": 81.16, "GEGEN": 99.87},
    {"dataset": "German",     "RBF": 75.36, "LINEAR": 76.87, "POLY": 76.86, "GCK": 74.83, "LEGEN": 71.33, "CHEB": 70.15, "HERMITE": 77.20, "GEGEN": 76.92},
    {"dataset": "Glass",      "RBF": 92.65, "LINEAR": 92.14, "POLY": 92.33, "GCK": 92.35, "LEGEN": 92.20, "CHEB": 89.49, "HERMITE": 92.14, "GEGEN": 92.19},
    {"dataset": "haberman",   "RBF": 74.81, "LINEAR": 73.42, "POLY": 74.21, "GCK": 74.43, "LEGEN": 74.62, "CHEB": 72.44, "HERMITE": 74.44, "GEGEN": 75.09},
    {"dataset": "Heart",      "RBF": 84.39, "LINEAR": 84.21, "POLY": 83.94, "GCK": 82.02, "LEGEN": 77.56, "CHEB": 72.72, "HERMITE": 85.26, "GEGEN": 85.98},
    {"dataset": "ionosphere", "RBF": 95.88, "LINEAR": 89.38, "POLY": 92.24, "GCK": 91.89, "LEGEN": 92.82, "CHEB": 69.22, "HERMITE": 94.12, "GEGEN": 93.37},
    {"dataset": "Liver",      "RBF": 74.25, "LINEAR": 69.62, "POLY": 72.80, "GCK": 74.71, "LEGEN": 73.08, "CHEB": 71.47, "HERMITE": 73.89, "GEGEN": 74.13},
    {"dataset": "Monks-1",    "RBF": 82.56, "LINEAR": 69.20, "POLY": 72.23, "GCK": 74.69, "LEGEN": 76.33, "CHEB": 68.99, "HERMITE": 85.75, "GEGEN": 97.52},
    {"dataset": "Monks-2",    "RBF": 81.95, "LINEAR": 62.15, "POLY": 71.77, "GCK": 77.75, "LEGEN": 85.03, "CHEB": 73.19, "HERMITE": 73.58, "GEGEN": 85.64},
    {"dataset": "Monks-3",    "RBF": 91.88, "LINEAR": 81.44, "POLY": 80.12, "GCK": 88.31, "LEGEN": 84.03, "CHEB": 73.42, "HERMITE": 93.42, "GEGEN": 93.46},
    {"dataset": "plrx",       "RBF": 73.40, "LINEAR": 71.46, "POLY": 71.97, "GCK": 72.43, "LEGEN": 71.53, "CHEB": 72.85, "HERMITE": 72.42, "GEGEN": 72.31},
    {"dataset": "sonar",      "RBF": 90.53, "LINEAR": 80.08, "POLY": 87.29, "GCK": 86.68, "LEGEN": 73.55, "CHEB": 70.21, "HERMITE": 89.91, "GEGEN": 92.25},
    {"dataset": "splice",     "RBF": 87.96, "LINEAR": 80.64, "POLY": 82.94, "GCK": 87.22, "LEGEN": 56.92, "CHEB": 51.83, "HERMITE": 92.26, "GEGEN": 93.62},
    {"dataset": "vehicle",    "RBF": 84.56, "LINEAR": 82.73, "POLY": 85.07, "GCK": 85.51, "LEGEN": 77.64, "CHEB": 75.09, "HERMITE": 85.05, "GEGEN": 85.24},
    {"dataset": "wdbc",       "RBF": 98.23, "LINEAR": 97.87, "POLY": 97.92, "GCK": 98.09, "LEGEN": 95.45, "CHEB": 95.72, "HERMITE": 98.35, "GEGEN": 98.15},
    {"dataset": "wpbc",       "RBF": 82.13, "LINEAR": 81.51, "POLY": 81.80, "GCK": 81.40, "LEGEN": 76.40, "CHEB": 76.54, "HERMITE": 82.11, "GEGEN": 82.56},
]

def load_expected_results() -> pd.DataFrame:
    """
    Devuelve un DataFrame con las métricas esperadas de precisión para cada
    par (dataset, kernel).
    Columnas: ['dataset', 'kernel', 'accuracy']
    """
    df = pd.DataFrame(_EXPECTED_RESULTS)
    df_long = df.melt(id_vars=['dataset'], var_name='kernel', value_name='accuracy')
    log.info(f"Cargados resultados esperados para {len(df_long)} combinaciones")
    return df_long

def list_available_libsvm(libsvm_dir: Optional[Path] = None) -> List[str]:
    """
    Return a list of dataset base-names available in the LIBSVM_DIR.
    If libsvm_dir is provided, it uses that path; otherwise uses default LIBSVM_DIR.
    """
    d = libsvm_dir or LIBSVM_DIR
    out: List[str] = []
    if not d.exists():
        log.warning(f"LIBSVM_DIR '{d}' does not exist.")
        return out
    for p in d.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in {'.txt', '.data', '.libsvm'}:
            out.append(p.stem)
    log.info(f"Found {len(out)} libsvm files in {d}")
    return out

def load_dataset(name: str) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Attempt to load a single dataset by name.

    Strategy:
    1) Search LIBSVM_DIR for files whose stem or filename contains the requested name (case-insensitive).
       If found, load with load_svmlight_file and return (name, X, y).
    2) If not found, raise FileNotFoundError so caller can fallback to load_all_datasets.
    """
    name_l = name.strip().lower()

    # 1) Try libsvm directory
    if LIBSVM_DIR.exists():
        for p in LIBSVM_DIR.iterdir():
            if not p.is_file():
                continue
            stem = p.stem.strip().lower()
            fname = p.name.strip().lower()
            # match exact or substring (allow "a1a" match "a1a.txt")
            if name_l == stem or name_l == fname or name_l in stem or name_l in fname:
                try:
                    Xs, y = load_svmlight_file(str(p), zero_based=True)
                    X = Xs.toarray()
                    log.info(f"Loaded LIBSVM dataset '{name}' from file {p} with shape {X.shape}")
                    return (name, X, y)
                except Exception as e:
                    log.warning(f"Failed to load {p} as LIBSVM: {e}")
                    # continue searching other files

    # 2) (Optional) Try to fetch from UCI by matching names in _EXPECTED_RESULTS (best-effort)
    #    Here we don't have a mapping to UCI IDs; so we attempt a best-effort strategy:
    #    if the requested name matches one of the expected dataset names, we try to fetch
    expected_names = {str(d['dataset']).strip().lower(): d for d in _EXPECTED_RESULTS}
    if name_l in expected_names:
        # No UCI id provided; user should rely on load_all_datasets with uci_list in config.
        log.warning(f"Dataset '{name}' is listed in EXPECTED_RESULTS but no automatic UCI id is available. "
                    "Falling back: caller should use load_all_datasets with the appropriate uci id in config.")
        raise FileNotFoundError(f"UCI single-dataset fetch not implemented for '{name}' (no id mapping).")

    # If nothing found:
    raise FileNotFoundError(f"Dataset '{name}' not found in LIBSVM_DIR ({LIBSVM_DIR}).")

def load_all_datasets(
    uci_list: List[Tuple[int, str]],
    libsvm_dir: Path,
    libsvm_files: List[Tuple[str, str]]
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Carga datasets según la configuración recibida:
      - uci_list: lista de (dataset_id, nombre)
      - libsvm_dir: carpeta donde están los .txt
      - libsvm_files: lista de (filename, nombre)
    """
    datasets = []
    # UCI
    for ds_id, name in uci_list:
        try:
            df = fetch_ucirepo(id=ds_id).data
            X = df.features.values; y = df.targets.values.ravel()
            mask = ~np.isnan(X).any(axis=1)
            X, y = X[mask], y[mask]
            log.info(f"UCI '{name}': {X.shape}")
            datasets.append((name, X, y))
        except Exception as e:
            log.warning(f"Failed to load UCI dataset id={ds_id}, name={name}: {e}")

    # LIBSVM (explicit list)
    for filename, name in libsvm_files:
        path = libsvm_dir / filename
        if not path.exists():
            log.warning(f"LIBSVM file not found: {path}")
            continue
        try:
            Xs, y = load_svmlight_file(str(path), zero_based=True)
            X = Xs.toarray()
            log.info(f"LIBSVM '{name}': {X.shape}")
            datasets.append((name, X, y))
        except Exception as e:
            log.warning(f"Failed to load LIBSVM file {path}: {e}")

    return datasets
