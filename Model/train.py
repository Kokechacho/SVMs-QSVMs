# model/train.py
from typing import Callable, Dict, Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import SVC
from collections import Counter


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

def train_svm(
    X, y,
    kernel_func,
    C=1.0,
    cv=10,
    n_jobs=-1,
    pca_dim: int | None = None,
    random_state: int | None = None
):
    """
    Realiza validación cruzada con el kernel dado.
    """
    # Montar pipeline
    steps = [('scaler', MinMaxScaler(feature_range=(-1, 1)))]
    if pca_dim is not None:
        steps.append(('pca', PCA(n_components=pca_dim)))
    steps.append(('svc', SVC(kernel=kernel_func, C=C, random_state=random_state)))
    pipe = Pipeline(steps)

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Listas para métricas
    support_props = []
    acc_val, acc_tr = [], []
    f1_val, f1_tr = [], []
    prec_val, prec_tr = [], []
    rec_val, rec_tr = [], []

    for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[test_idx], y[test_idx]

        # Entrenamiento
        pipe.fit(X_tr, y_tr)

        # Predicciones
        y_pred_val = pipe.predict(X_val)
        y_pred_tr  = pipe.predict(X_tr)

        # Accuracy
        acc = pipe.score(X_val, y_val)
        acc_val.append(acc)
        acc_tr.append(pipe.score(X_tr, y_tr))

        # Proporción de vectores de soporte en train
        n_sup = pipe.named_steps['svc'].support_.shape[0]
        support_props.append(n_sup / len(X_tr) * 100)
        # support_props.append(n_sup / acc * 100)

        # F1, Precision, Recall con zero_division=0 para evitar warnings
        f1_val.append(f1_score(y_val, y_pred_val, average='weighted', zero_division=0))
        f1_tr.append(f1_score(y_tr, y_pred_tr, average='weighted', zero_division=0))
        
        prec_val.append(precision_score(y_val, y_pred_val, average='weighted', zero_division=0))
        prec_tr.append(precision_score(y_tr, y_pred_tr, average='weighted', zero_division=0))
        
        rec_val.append(recall_score(y_val, y_pred_val, average='weighted', zero_division=0))
        rec_tr.append(recall_score(y_tr, y_pred_tr, average='weighted', zero_division=0))

    # Agregar estadísticas
    stats = {
        'accuracy':               np.mean(acc_val) * 100,
        'accuracy_std':           np.std(acc_val) * 100,
        'train_accuracy':         np.mean(acc_tr) * 100,
        'train_accuracy_std':     np.std(acc_tr) * 100,
        'f1_score':               np.mean(f1_val) * 100,
        'f1_score_std':           np.std(f1_val) * 100,
        'train_f1_score':         np.mean(f1_tr) * 100,
        'train_f1_score_std':     np.std(f1_tr) * 100,
        'precision':              np.mean(prec_val) * 100,
        'precision_std':          np.std(prec_val) * 100,
        'train_precision':        np.mean(prec_tr) * 100,
        'train_precision_std':    np.std(prec_tr) * 100,
        'recall':                 np.mean(rec_val) * 100,
        'recall_std':             np.std(rec_val) * 100,
        'train_recall':           np.mean(rec_tr) * 100,
        'train_recall_std':       np.std(rec_tr) * 100,
        'support_vector_acc':     np.mean(support_props),
        'support_vector_std':     np.std(support_props)
    }
    return stats


def tune_svm(
    X: np.ndarray,
    y: np.ndarray,
    kernel_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    param_grid: Dict,
    cv: int = 5,
    n_jobs: int = -1
) -> BaseEstimator:
    """
    Realiza búsqueda de cuadrícula (GridSearchCV) para SVM con kernel custom.

    :param X: Features de entrenamiento
    :param y: Labels
    :param kernel_func: Kernel callable
    :param param_grid: Diccionario de parámetros para GridSearchCV
    :param cv: folds
    :param n_jobs: cores
    :return: Mejor estimador
    """
    from sklearn.svm import SVC

    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('svc', SVC(kernel=kernel_func))
    ])
    grid = GridSearchCV(
        estimator=pipe,
        param_grid={f'svc__{k}': v for k, v in param_grid.items()},
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs
    )
    grid.fit(X, y)
    print(f"Best params: {grid.best_params_}, Best score: {grid.best_score_*100:.2f}%")
    return grid.best_estimator_
