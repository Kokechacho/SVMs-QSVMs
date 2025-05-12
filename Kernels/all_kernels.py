from typing import List, Dict, Callable
import Kernels.base as b
import Kernels.custom as c

def build_kernels(
    hermite_degree: int = 6,
    gegen_degree: int = 3,
    gegen_alpha: float = 0.1
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
        'name': f'HERMITE',
        'func': lambda X, Y, n=hermite_degree: c.hermite_kernel_matrix(X, Y, n)
    })
    # Gegenbauer
    kernels.append({
        'name': 'GEGEN',
        'func': lambda X, Y, n=gegen_degree, alpha=gegen_alpha: c.gegenbauer_kernel(X, Y, n, alpha)
    })
    return kernels