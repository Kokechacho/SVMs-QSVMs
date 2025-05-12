# model/train.py
from typing import Callable, Dict, Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    kernel_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    C: float = 1.0,
    cv: int = 5,
    n_jobs: int = -1,
    **kernel_params
) -> BaseEstimator:
    """
    Entrena un SVM usando Pipeline con escalado e inyección de kernel custom.

    :param X: Entrenamiento features (n_samples, n_features)
    :param y: Entrenamiento labels
    :param kernel_func: Callable que computa la matriz de Gram (custom kernel)
    :param C: Parámetro de regularización
    :param cv: Número de folds para validación cruzada
    :param n_jobs: Paralelización
    :param kernel_params: Parámetros adicionales (passed via SVC)
    :return: SVC entrenado con mejores parámetros
    """
    from sklearn.svm import SVC

    # Pipeline con escalador y SVM con custom kernel
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('svc', SVC(kernel=kernel_func, C=C, **kernel_params))
    ])

    # Validación cruzada para accuracy y f1
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=n_jobs)
    print(f"CV Accuracy: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

    # Ajuste final sobre todo el training set
    pipe.fit(X, y)
    return pipe


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
