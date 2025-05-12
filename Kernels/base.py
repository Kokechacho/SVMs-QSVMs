from typing import Callable, Optional
import numpy as np
from sklearn.metrics.pairwise import linear_kernel as _linear
from sklearn.metrics.pairwise import polynomial_kernel as _poly
from sklearn.metrics.pairwise import rbf_kernel as _rbf


def linear() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Kernel lineal: k(x, y) = x · y
    :return: función que computa la matriz de Gram
    """
    return _linear


def polynomial(degree: int = 3,
               gamma: Optional[float] = None,
               coef0: float = 1.0) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Kernel polinómico: (gamma <x, y> + coef0)^degree
    :param degree: grado del polinomio
    :param gamma: escala de la inner product (si None usa 1/n_features)
    :param coef0: término independiente
    :return: función de kernel polinómico
    """
    return lambda X, Y: _poly(X, Y, degree=degree, gamma=gamma, coef0=coef0)


def rbf(gamma: Optional[float] = None) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Kernel RBF (Gaussiano): exp(-gamma ||x-y||^2)
    :param gamma: coeficiente (si None usa 1/n_features)
    :return: función de kernel RBF
    """
    return lambda X, Y: _rbf(X, Y, gamma=gamma)