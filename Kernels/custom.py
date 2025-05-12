from typing import Callable
import numpy as np
from sklearn.metrics.pairwise import check_pairwise_arrays


def compute_he(x: np.ndarray, max_order: int) -> np.ndarray:
    """
    Computa polinomios de Hermite probabilistas He_0 .. He_max_order
    :param x: vector (n_samples,)
    :param max_order: grado máximo
    :return: matriz (n_samples, max_order+1)
    """
    x = np.asarray(x).ravel()
    n_samples = x.shape[0]
    He = np.zeros((n_samples, max_order + 1), dtype=float)
    He[:, 0] = 1
    if max_order >= 1:
        He[:, 1] = x
    for i in range(1, max_order):
        He[:, i+1] = x * He[:, i] - i * He[:, i-1]
    return He


def hermite_kernel_matrix(X: np.ndarray, Y: np.ndarray, n: int) -> np.ndarray:
    """
    Kernel de Hermite probabilista vectorizado:
      K(x,z) = ∏_{j=0..d-1} [ ∑_{i=0}^n 2^{-2i} He_i(x_j) He_i(z_j) * exp(-(x_j^2+z_j^2)/2) ]
    :param X: array (m, d)
    :param Y: array (p, d)
    :param n: grado máximo de Hermite
    :return: matriz (m, p)
    """
    X, Y = check_pairwise_arrays(X, Y)
    m, d = X.shape
    p, _ = Y.shape

    # Precomputar polinomios de Hermite para todas las características
    H_X = np.zeros((m, d, n+1))
    H_Y = np.zeros((p, d, n+1))
    for j in range(d):
        H_X[:, j, :] = compute_he(X[:, j], n)
        H_Y[:, j, :] = compute_he(Y[:, j], n)

    # Factores de escalado !!!!!!!!!!!!!!!!!!
    # En el paper usan el factor de escalado en función de n pero eso funciona muy mal
    scales = np.array([2**(-2 * i) for i in range(n+1)])
    # scales = np.array([2**(-2 * n) for i in range(n+1)])

    # Términos exponenciales
    X_exp = np.exp(-X**2 / 2)
    Y_exp = np.exp(-Y**2 / 2)

    # Inicializar matriz del kernel
    K = np.ones((m, p))

    for j in range(d):
        # Extraemos las matrices univariadas para la coordenada j:
        #  H_Xj[a,i] = He_i(X[a,j])
        #  H_Yj[b,i] = He_i(Y[b,j])
        H_Xj = H_X[:, j, :]    
        H_Yj = H_Y[:, j, :]    

        # — SUMATORIO sobre i:  ∑_{i=0}^n scales[i] * He_i(x_j) * He_i(z_j)
        # K_j[a,b] = ∑_i  (H_Xj[a,i] * scales[i]) * H_Yj[b,i]
        K_j = np.dot(H_Xj * scales, H_Yj.T)

        # — FACTOR GAUSSIANO unidimensional: exp(−(x_j^2 + z_j^2)/2)
        # exp_j[a,b] = exp(− X[a,j]^2/2 ) * exp(− Y[b,j]^2/2 )
        exp_j = np.outer(X_exp[:, j], Y_exp[:, j])

        # — PRODUCTORIO sobre j: 
        # actualizamos K[a,b] *= (K_j[a,b] * exp_j[a,b])
        K *= K_j * exp_j

    return K


def compute_gegenbauer(x: np.ndarray, n_max: int, alpha: float) -> np.ndarray:
    """
    Genera polinomios de Gegenbauer C_0..C_n_max(M) en cada x
    :param x: vector (n_samples,)
    :param n_max: grado máximo
    :param alpha: parámetro
    :return: matriz (n_samples, n_max+1)
    """
    x = np.asarray(x).ravel()
    m = x.shape[0]
    C = np.zeros((m, n_max+1), dtype=float)
    C[:, 0] = 1
    if n_max >= 1:
        C[:, 1] = 2 * alpha * x
    for n in range(1, n_max):
        C[:, n+1] = ((2*(n+alpha)*x*C[:, n] - (n+2*alpha-1)*C[:, n-1])/(n+1))
    return C


def gegenbauer_weight(x: np.ndarray, z: np.ndarray, alpha: float, epsilon: float=0.1) -> np.ndarray:
    """
    Peso w_α(x,z) = [(1-x^2)(1-z^2)]^(α-1/2) + ε
    """
    x = np.asarray(x).ravel()
    z = np.asarray(z).ravel()
    m, p = x.shape[0], z.shape[0]
    if -0.5 < alpha <= 0.5:
        return np.ones((m, p), dtype=float)
    term_x = np.clip(1-x**2, 0, None)
    term_z = np.clip(1-z**2, 0, None)
    product = term_x[:, None] * term_z[None, :]
    return product**(alpha-0.5) + epsilon


def compute_u(i: int, alpha: float) -> float:
    """
    Factor u(C_i) = 1 / [sqrt(i+1) * |C_i^α(1)|]
    """
    C_at_1 = compute_gegenbauer(np.array([1.0]), i, alpha)[0, i]
    denom = np.sqrt(i+1) * abs(C_at_1)
    if denom == 0:
        raise ValueError("C_i^α(1) es cero")
    return 1.0/denom


def gegenbauer_kernel(X: np.ndarray, Z: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """
    Kernel de Gegenbauer:
      K(x,z) = ∏_{j=1..d} [ ∑_{i=0..n} C_i^α(x_j)C_i^α(z_j) * u_i^2 * w_α(x_j,z_j) ]
    """
    X, Z = check_pairwise_arrays(X, Z)
    m, d = X.shape
    p, _ = Z.shape
    C_X = np.zeros((m, d, n+1))
    C_Z = np.zeros((p, d, n+1))
    for j in range(d):
        C_X[:, j, :] = compute_gegenbauer(X[:, j], n, alpha)
        C_Z[:, j, :] = compute_gegenbauer(Z[:, j], n, alpha)
    u_sq = np.array([compute_u(i, alpha)**2 for i in range(n+1)])
    W = np.zeros((m, p, d), dtype=float)
    for j in range(d):
        W[:, :, j] = gegenbauer_weight(X[:, j], Z[:, j], alpha)
    K = np.ones((m, p), dtype=float)
    for j in range(d):
        Cj_X = C_X[:, j, :]
        Cj_Z = C_Z[:, j, :]
        sum_i = (Cj_X * u_sq).dot(Cj_Z.T)
        K *= sum_i * W[:, :, j]
    return K
