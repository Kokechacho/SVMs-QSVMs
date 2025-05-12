
from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import ClassifierMixin


def evaluate_model(
    model: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evalúa un modelo SVM entrenado en datos de test.

    :param model: Pipeline o SVC entrenado
    :param X_test: Features de test escalados o pipeline que incluye scaler
    :param y_test: Labels verdaderas
    :return: dict con 'accuracy', 'f1_score', 'n_support_vectors'
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100
    # Extraer número total de vectores soporte
    try:
        n_support = sum(model.named_steps['svc'].n_support_)
    except Exception:
        n_support = 0
    return {
        'accuracy': acc,
        'f1_score': f1,
        'n_support_vectors': n_support
    }
