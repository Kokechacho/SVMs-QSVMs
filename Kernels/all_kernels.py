# Kernels/all_kernels.py
from typing import List, Dict, Callable, Optional
import Kernels.base as base
import Kernels.custom as custom
import numpy as np

def build_kernels(
    poly_degree: int = 3,
    poly_gamma: str = "scale",
    rbf_gamma: str = "scale", 
    hermite_degree: int = 6,
    gegen_degree: int = 3,
    gegen_alpha: float = 0.1,
    al_degree: int = 3,
    al_a: float = -1,
    al_q: float = 0.5,
    specific_kernels: Optional[List[str]] = None
) -> List[Dict[str, Callable]]:
    """
    Construye kernels con parámetros DINÁMICOS para optimización
    """
    kernels = []
    
    kernel_map = {
        'linear': {'name': 'LINEAR', 'func': base.linear()},
        'poly': {
            'name': 'POLY', 
            'func': base.polynomial(degree=poly_degree, gamma=poly_gamma)
        },
        'rbf': {
            'name': 'RBF', 
            'func': base.rbf(gamma=rbf_gamma)
        },
        'custom_hermite': {
            'name': 'HERMITE',
            'func': lambda X, Y: custom.hermite_kernel_matrix(X, Y, hermite_degree)
        },
        'custom_gegen': {
            'name': 'GEGEN',
            'func': lambda X, Y: custom.gegenbauer_kernel(X, Y, gegen_degree, gegen_alpha)
        },
        'custom_alsalam': {
            'name': 'AL-SALAM', 
            'func': lambda X, Y: custom.kernel_AlSalam(X, Y, al_q, al_a, al_degree)
        }
    }
    
    if specific_kernels:
        for kernel_name in specific_kernels:
            if kernel_name in kernel_map:
                kernels.append(kernel_map[kernel_name])
    else:
        kernels = list(kernel_map.values())
    
    return kernels

def build_single_kernel(kernel_name: str, **kwargs):
    """Construye un solo kernel con parámetros específicos"""
    kernel_map = {
        'linear': {'name': 'LINEAR', 'func': base.linear()},
        'poly': {
            'name': 'POLY', 
            'func': base.polynomial(
                degree=kwargs.get('degree', 3), 
                coef0=kwargs.get('coef0', 1.0)  
            )
        },
        'rbf': {
            'name': 'RBF', 
            'func': base.rbf(gamma=kwargs.get('gamma', "scale"))
        },
        'custom_hermite': {
            'name': 'HERMITE',
            'func': lambda X, Y: custom.hermite_kernel_matrix(X, Y, kwargs.get('degree', 6))
        },
        'custom_gegen': {
            'name': 'GEGEN',
            'func': lambda X, Y: custom.gegenbauer_kernel(X, Y, kwargs.get('degree', 3), 
                                                 kwargs.get('alpha', 0.1))
        },
        'custom_alsalam': {
            'name': 'AL-SALAM', 
            'func': lambda X, Y: custom.kernel_AlSalam(X, Y, kwargs.get('q', 0.5), 
                                              kwargs.get('a', -1), 
                                              kwargs.get('degree', 3))
        }
    }
    
    return kernel_map.get(kernel_name)