# model/train.py
from typing import Callable, Dict, Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.svm import SVC


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
    pca_dim: int | None = None
):
    """
    Realiza validación cruzada con el kernel dado.
    Si pca_dim!=None, aplica PCA(n_components=pca_dim) dentro del pipeline
    (previo al escalado y al SVC), fold a fold.
    """
    steps = [ ('scaler', MinMaxScaler(feature_range=(-1, 1)))]
    if pca_dim is not None:
        steps.append(('pca', PCA(n_components=pca_dim)))
    steps += [
        ('svc',    SVC(kernel=kernel_func, C=C))
    ]

    pipe = Pipeline(steps)
    cv_splitter = StratifiedKFold(n_splits=cv)
    support_props, accs, f1s = [], [], []

    for train_idx, test_idx in cv_splitter.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        pipe.fit(X_tr, y_tr)

        n_sup = pipe.named_steps['svc'].support_.shape[0]
        support_props.append(n_sup / len(X_tr) * 100)

        X_val, y_val = X[test_idx], y[test_idx]
        accs.append(pipe.score(X_val, y_val))
        f1s.append(f1_score(y_val, pipe.predict(X_val), average='weighted'))

    return {
        'accuracy':          np.mean(accs) * 100,
        'accuracy_std':      np.std(accs) * 100,
        'f1_score':          np.mean(f1s) * 100,
        'f1_score_std':      np.std(f1s) * 100,
        'support_vector_acc': np.mean(support_props),
        'support_vector_std': np.std(support_props)
    }


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
