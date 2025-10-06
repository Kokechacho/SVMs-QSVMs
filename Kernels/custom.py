from typing import Callable
import numpy as np
from sklearn.metrics.pairwise import check_pairwise_arrays
from scipy.special import eval_gegenbauer


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

def compute_he(x: np.ndarray, max_order: int) -> np.ndarray:
    """
    Hermite probabilistas He_0..He_max_order, vectorizado.
    """
    x = np.asarray(x).ravel()
    n_samples = x.shape[0]
    He = np.zeros((n_samples, max_order + 1), dtype=float)
    He[:, 0] = 1
    if max_order >= 1:
        He[:, 1] = x
    for i in range(1, max_order):
        He[:, i + 1] = x * He[:, i] - i * He[:, i - 1]
    return He


def hermite_kernel_matrix_fast(X: np.ndarray, Y: np.ndarray, n: int) -> np.ndarray:
    """
    Versión optimizada del kernel de Hermite probabilista:
        K(x,z) = ∏_j [ ∑_i 2^{-2i} He_i(x_j) He_i(z_j) * exp(-(x_j^2+z_j^2)/2) ]
    """
    X, Y = check_pairwise_arrays(X, Y)
    m, d = X.shape
    p, _ = Y.shape

    scales = 2 ** (-2 * np.arange(n + 1))
    X_exp = np.exp(-X ** 2 / 2)
    Y_exp = np.exp(-Y ** 2 / 2)

    K = np.ones((m, p), dtype=float)

    # Procesamos una dimensión a la vez (evita gran tensor 3D)
    for j in range(d):
        He_X = compute_he(X[:, j], n)  # (m, n+1)
        He_Y = compute_he(Y[:, j], n)  # (p, n+1)

        # Vectorizamos: (He_X * scales) @ He_Y.T == Σ_i scales[i] He_i(x_j)He_i(z_j)
        K_j = (He_X * scales) @ He_Y.T

        # Factor gaussiano
        exp_j = np.outer(X_exp[:, j], Y_exp[:, j])

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
    u_sq = np.array([compute_u(i, alpha)**2 if alpha > 0.5 else 1 for i in range(n+1)])
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

def gegenbauer_kernel_fast(X: np.ndarray, Z: np.ndarray, n: int, alpha: float, eps: float = 1e-8) -> np.ndarray:
    """
    Vectorized Gegenbauer kernel:
      K(x,z) = ∏_{j=1..d} [ ∑_{i=0..n} C_i^α(x_j) * C_i^α(z_j) * u_i^2 * w_α(x_j, z_j) ]
    """

    X, Z = check_pairwise_arrays(X, Z)
    X = np.clip(X, -1.0, 1.0)
    Z = np.clip(Z, -1.0, 1.0)

    m, d = X.shape
    p, _ = Z.shape

    # Compute all Gegenbauer polynomials at once
    degrees = np.arange(0, n + 1)
    C_X = np.stack([eval_gegenbauer(deg, alpha, X) for deg in degrees], axis=-1)  # shape (m, d, n+1)
    C_Z = np.stack([eval_gegenbauer(deg, alpha, Z) for deg in degrees], axis=-1)  # shape (p, d, n+1)

    # Compute normalization factors u_i
    if alpha > 0.5:
        C1_vals = eval_gegenbauer(degrees, alpha, 1.0)
        u_sq = 1.0 / ((np.sqrt(degrees + 1) * np.abs(C1_vals))**2)
    else:
        u_sq = np.ones_like(degrees, dtype=float)

    # Compute weights (vectorized)
    if alpha > 0.5:
        term_x = np.clip(1 - X**2, 0, None)
        term_z = np.clip(1 - Z**2, 0, None)
        W = (term_x[:, None, :] * term_z[None, :, :])**(alpha - 0.5) + eps
    else:
        W = np.ones((m, p, d), dtype=float)

    # Compute the kernel (vectorized over j and i)
    # (m, p, d) ← sum_i (C_X[:,:,i] * u_sq[i]) @ (C_Z[:,:,i]).T
    K = np.ones((m, p), dtype=float)
    for j in range(d):
        CXj = C_X[:, j, :] * u_sq[None, :]
        CZj = C_Z[:, j, :]
        S = CXj @ CZj.T
        K *= S * W[:, :, j]

    return K

def gegenbauer_kernel(X, Z, n, alpha):
    return gegenbauer_kernel_fast(X, Z, n, alpha)

def compute_al_salam_carlitz_U(x: np.ndarray, q: float, a: float, n_max: int) -> np.ndarray:
    """
    Calcula los polinomios U_0..U_n_max de Al-Salam–Carlitz I evaluados en cada valor de x.
    Utiliza la recurrencia:
        U_{-1}(x) = 0, 
        U_0(x) = 1,
        U_1(x) = x - (a + 1),
        U_{n+1}(x) = (x - (a + 1) q^n) * U_n(x) + a q^{n-1} (1 - q^n) * U_{n-1}(x)
        
    :param x: numpy array, shape (m,)
    :param q: parámetro q
    :param a: parámetro a
    :param n_max: grado máximo
    :return: matriz numpy shape (m, n_max+1) con U_0 ... U_n_max para cada x
    """
    x = np.asarray(x).ravel()
    m = x.shape[0]
    
    # Matriz de salida: filas=muestras, columnas=U_0 a U_n_max
    U = np.zeros((m, n_max + 1), dtype=float)
    
    # Condiciones iniciales
    U[:, 0] = 1.0                       # U_0(x) = 1
    if n_max >= 1:
        U[:, 1] = x - (a + 1)           # U_1(x) = x - (a+1)
    
    # Recurrencia para n >= 1
    for n in range(1, n_max):
        qn = q ** n
        qnm1 = q ** (n - 1)
        U[:, n + 1] = ((x - (a + 1) * qn) * U[:, n] +
                       a * qnm1 * (1 - qn) * U[:, n - 1])
    
    return U

# ---------------------------------------------------------------------------------------------
# Función auxiliar para calcular el operador de Pochhammer cuando tiende a infinito
# La necesitamos para calcular la función de peso posterior
# ---------------------------------------------------------------------------------------------

def q_pochhammer_inf(z, q, n_terms=500):
    """
    Aproxima (z; q)_∞ ≈ ∏_{k=0}^{n_terms-1} (1 - z * q^k).
    """
    z = np.asarray(z, dtype=float)
    result = np.ones_like(z)
    qk = 1.0
    for _ in range(n_terms):
        result *= (1 - z * qk)
        qk *= q
    return result


# ---------------------------------------------------------------------------------------------
# FUNCIÓN DE PESO DE LOS AL-SALAM CARLIZT TIPO I 
# ---------------------------------------------------------------------------------------------

def weight_al_salam_carlitz(x, q, a, n_terms=400):
    """
    Función de peso para polinomios Al-Salam–Carlitz I:
    
        w(x; a, q) = (q*x; q)_∞ * (a^{-1}*q*x; q)_∞,
    
    aproximando cada Pochhammer infinito con n_terms factores.
    
    :param x: array o escalar de puntos
    :param q: parámetro q (0 < q < 1)
    :param a: parámetro a (no nulo)
    :param n_terms: número de términos para truncar los productos
    :return: array o escalar w(x)
    """
    x = np.asarray(x, dtype=float)
    w1 = q_pochhammer_inf(q * x, q, n_terms)
    w2 = q_pochhammer_inf((q * x) / a, q, n_terms)
    return w1 * w2

def scaling_factor_al_salam_carlitz(a, q, i, N):
    """
    Devuelve el factor de escalado:
        scale = 1 / (sqrt(N+1) * |U_i^a(-1; q)|)
    
    :param a: parámetro a
    :param q: parámetro q
    :param i: grado del polinomio U_i
    :return: escala (float)
    """
    U = compute_al_salam_carlitz_U(-1, q, a, i)
    U_i_abs = np.abs(U[0, i])
    
    if U_i_abs == 0:
        return 0.0  # evitar división por cero
    return 1.0 / (np.sqrt(N + 1) * U_i_abs)


def kernel_AlSalam(X: np.ndarray, Z: np.ndarray, q: float, a: float, N: int) -> np.ndarray:
    """
    Kernel de productos de polinomios Al-Salam–Carlitz I con escalado y peso.

    K(x, z) = ∏_{dim=1}^d
      ⟨ φ_dim(x), φ_dim(z) ⟩

    donde φ_dim(t) es el vector de características en la dimensión 'dim':
      φ_dim(t)[i] = U_i(t; q, a) * scale_i * w(t; q, a)

    con:
     - U_i(t; q, a): polinomio de grado i evaluado en t
     - scale_i = 1 / (√(i+1) · |U_i(-1; q, a)|)^2
     - w(t; q, a): función de peso

    :param X: matriz de datos de entrenamiento, forma (nX, d)
    :param Z: matriz de datos de prueba, forma (nZ, d)
    :param q: parámetro q de los polinomios
    :param a: parámetro a de los polinomios
    :param N: grado máximo de polinomios
    :return: matriz del kernel K de forma (nX, nZ)
    """
    nX, d = X.shape
    nZ, _ = Z.shape

    # Inicializamos el kernel como 1 (producto acumulado por dimensión)
    K = np.ones((nX, nZ), dtype=float)

    # Precompute weights once
    weights_X = [np.maximum(weight_al_salam_carlitz(X[:, j], q, a, 400), 0.1) for j in range(d)]
    weights_Z = [np.maximum(weight_al_salam_carlitz(Z[:, j], q, a, 400), 0.1) for j in range(d)]

    # Precomputamos los factores de escalado scale_i para i=0..N
    # scales = np.array([scaling_factor_al_salam_carlitz(a, q, i, N) for i in range(N + 1)])**2
    # scales[i] = 1 / (√(i+1) · |U_i(-1; q,a)|)**2

    for dim in range(d):
        # 1) Evaluamos los polinomios U_0..U_N en toda la dimensión dim
        Ux = compute_al_salam_carlitz_U(X[:, dim:dim+1].ravel(), q, a, N)  # (nX, N+1)
        Uz = compute_al_salam_carlitz_U(Z[:, dim:dim+1].ravel(), q, a, N)  # (nZ, N+1)

        # 2) Función de peso w(t) para esa dimensión
        phi_x = Ux * weights_X[dim][:, None]
        phi_z = Uz * weights_Z[dim][:, None]

        # 3) Producto interno entre φ_x y φ_z para esta dimensión
        K_dim = phi_x @ phi_z.T   # (nX, nZ)

        # 4) Acumulamos por producto en cada dimensión
        K *= K_dim

    return K