# model/train.py
from typing import Callable, Dict, Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.svm import SVC


def train_svm(X, y, kernel_func, C=1.0, cv=10, n_jobs=-1):
    """
    Realiza validación cruzada con el kernel dado y devuelve métricas promedio,
    incluyendo estadísticas de vectores soporte.
    """
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('svc', SVC(kernel=kernel_func, C=C))
    ])
    
    # Configurar cross-validation
    cv_splitter = StratifiedKFold(n_splits=cv)
    support_proportions = []
    acc_scores = []
    f1_scores = []
    
    for train_idx, _ in cv_splitter.split(X, y):
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        
        # Entrenar modelo en el fold
        pipe.fit(X_train_fold, y_train_fold)
        
        # Obtener porcentaje de vectores soporte
        n_support = pipe.named_steps['svc'].support_.shape[0]
        support_prop = (n_support / len(X_train_fold)) * 100  # Porcentaje
        support_proportions.append(support_prop)
        
        # Calcular métricas en validation
        X_val_fold, y_val_fold = X[~train_idx], y[~train_idx]
        acc = pipe.score(X_val_fold, y_val_fold)
        f1 = f1_score(y_val_fold, pipe.predict(X_val_fold), average='weighted')
        
        acc_scores.append(acc)
        f1_scores.append(f1)
    
    # Convertir a arrays numpy
    acc_scores = np.array(acc_scores)
    f1_scores = np.array(f1_scores)
    support_proportions = np.array(support_proportions)
    
    return {
        'accuracy': acc_scores.mean() * 100,
        'accuracy_std': acc_scores.std() * 100,
        'f1_score': f1_scores.mean() * 100,
        'f1_score_std': f1_scores.std() * 100,
        'support_vector_acc': support_proportions.mean(),
        'support_vector_std': support_proportions.std()
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
