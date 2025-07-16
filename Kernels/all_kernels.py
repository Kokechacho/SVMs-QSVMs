from typing import List, Dict, Callable
import Kernels.base as b
import Kernels.custom as c
import numpy as np

def build_kernels(
    hermite_degree: int = 6,
    gegen_degree: int = 3,
    gegen_alpha: float = 0.1,
    al_degree: int = 3,
    al_a: float = -1.5,
    al_q: float = 0.5
) -> List[Dict[str, Callable]]:
    """
    Construye lista de kernels con nombre y función.
    """
    kernels = [
        {'name': 'LINEAR', 'func': b.linear()},
        {'name': 'POLY',   'func': b.polynomial(degree=3)},
        {'name': 'RBF',    'func': b.rbf()},
    ]
    # Hermite variantes
    kernels.append({
        'name': 'HERMITE',
        'func': lambda X, Y, n=hermite_degree: c.hermite_kernel_matrix(X, Y, n)
    })
    # Gegenbauer
    kernels.append({
        'name': 'GEGEN',
        'func': lambda X, Y, n=gegen_degree, alpha=gegen_alpha: c.gegenbauer_kernel(X, Y, n, alpha)
    })
    x_vals = np.linspace(-1, 1, 1000)
    z_vals = np.linspace(-1, 1, 1000)
    X, Z = np.meshgrid(x_vals, z_vals)

    weights = c.weight_al_salam_carlitz(X, al_q, al_a) * c.weight_al_salam_carlitz(Z, al_q, al_a)
    w = np.max(weights)
    kernels.append({
        'name': 'AL-SALAM',
        'func': lambda X, Y, n=al_degree, a=al_a, q=al_q: c.kernel_AlSalam(X, Y, q, a, n, w)
    })
    return kernels