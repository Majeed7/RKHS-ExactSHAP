# the implementations before moving to an indepdent esp comptuations

import numpy as np
from functools import reduce
from sklearn.metrics.pairwise import rbf_kernel
import math

import numpy as np
from numba import njit, prange

import numpy as np
from numba import njit, prange

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from esp import ESPComputer


'''
it solves the Mobius representation of Shapley values in RKHS with product kernel with Gauss Legendre quadrature
'''

## Numpy version
import numpy as np

def weighted_values_gauss_shared(K, alpha, m_q=None, rule="legendre"):
    """
    Approximate ∑_m K[i,m] * alpha[m] * ∫_0^1 ∏_{j≠i} (1 + x*K[j,m]) dx  for all i,
    using shared Gauss nodes and log-space products (no ESPs, no prefix/suffix).
    K: (d, m)  kernel vectors (k_j) per feature j and column m
    alpha: (m,)
    Exact for degree (d-1) if m_q >= ceil(d/2).
    """
    K = np.asarray(K, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    d, m = K.shape
    if m_q is None:
        m_q = (d + 1) // 2  # ceil(d/2)

    # Quadrature nodes/weights on [0,1]
    if rule == "legendre":
        x, w = np.polynomial.legendre.leggauss(m_q)  # [-1,1]
        x = 0.5*(x + 1.0)
        w = 0.5*w
    elif rule == "clenshaw":
        # Simple Clenshaw–Curtis (Chebyshev–Lobatto) nodes/weights on [0,1]
        k = np.arange(m_q)
        x = 0.5*(1 - np.cos(np.pi * k/(m_q-1)))
        # O(m_q^2) weights (fine for m_q<=256)
        w = np.zeros(m_q)
        for j in range(m_q):
            s = 1.0 if j in (0, m_q-1) else 2.0
            for r in range(1, (m_q-1)//2 + 1):
                s -= 2.0*np.cos(2*np.pi*r*j/(m_q-1)) / (4*r*r - 1)
            if (m_q-1) % 2 == 0:
                r = (m_q-1)//2
                s -= np.cos(2*np.pi*r*j/(m_q-1)) / (4*r*r - 1)
            w[j] = 2.0*s/(m_q-1)
        # normalize to integrate constants exactly
        w *= 1.0/np.sum(w)
    else:
        raise ValueError("rule must be 'legendre' or 'clenshaw'")

    # --- Shared product across features PER node&column, in log-space ---
    # We stream over features to avoid a (m_q,d,m) tensor in memory.
    log_abs_P = np.zeros((m_q, m), dtype=np.float64)
    sign_P = np.ones((m_q, m), dtype=np.float64)
    for j in range(d):
        t = 1.0 + np.outer(x, K[j, :])            # shape (m_q, m)
        sign_P *= np.sign(t)
        log_abs_P += np.log(np.abs(t), dtype=np.float64)

    # --- Per-i division and integration ---
    # result[i] = sum_m K[i,m] * alpha[m] * sum_ell w[ell] * (P / (1 + x*K[i,m]))
    result = np.zeros(d, dtype=np.float64)
    wa = w[:, None]                               # (m_q,1)
    a = alpha[None, :]                            # (1,m)
    for i in range(d):
        denom = 1.0 + np.outer(x, K[i, :])        # (m_q, m)
        integrand_sign = sign_P * np.sign(denom)
        integrand_log = log_abs_P - np.log(np.abs(denom), dtype=np.float64)
        Qint = (wa * (integrand_sign * np.exp(integrand_log))).sum(axis=0)  # (m,)
        result[i] = (K[i, :] * a * Qint).sum()
    return result

def weighted_values_gauss_legendre(K, alpha, m_q=None):
    """
    Exact for degree (d-1) polynomials if m_q >= ceil(d/2).
    K: (d, m)  kernel vectors (k_j)
    alpha: (m,)
    """
    K = np.asarray(K, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    d, m = K.shape
    if m_q is None:
        m_q = (d + 1) // 2  # ceil(d/2)

    # Gauss-Legendre nodes/weights on [-1,1], map to [0,1]
    x, w = np.polynomial.legendre.leggauss(m_q)   # exact for deg <= 2*m_q-1 on [-1,1]
    x = 0.5 * (x + 1.0)                           # to [0,1]
    w = 0.5 * w

    acc_vec = np.zeros((d, m), dtype=K.dtype)     # accumulate ∫ Q_i(x) dx elementwise
    # shapes: K (d,m), x (m_q,), w (m_q,)
    X = x[:, None, None]                  # (m_q,1,1)
    B = 1.0 + X * K[None, :, :]           # (m_q, d, m)

    # prefix/suffix per node independently (axis=1 is features)
    pref = np.cumprod(B, axis=1)
    pref = np.concatenate(
        [np.ones((m_q,1,m), B.dtype), pref[:, :-1, :]],
        axis=1
    )

    suf = np.cumprod(B[:, ::-1, :], axis=1)[:, ::-1, :]
    suf = np.concatenate(
        [suf[:, 1:, :], np.ones((m_q,1,m), B.dtype)],
        axis=1
    )

    Q_no_i = pref * suf                   # (m_q, d, m)
    acc_vec = (w[:, None, None] * Q_no_i).sum(axis=0)  # (d, m)

    # vector version
    # for t in range(m_q):
    #     xt = x[t]; wt = w[t]
    #     B = 1.0 + xt * K                          # (d, m)

    #     # exclusive product across features (no division), per quadrature node
    #     pref = np.cumprod(B, axis=0)
    #     pref = np.vstack([np.ones((1, m)), pref[:-1]])
    #     suf  = np.cumprod(B[::-1], axis=0)[::-1]
    #     suf  = np.vstack([suf[1:], np.ones((1, m))])

    #     Q_no_i = pref * suf                       # (d, m): ∏_{j≠i}(1+xt*k_j)
    #     acc_vec += wt * Q_no_i

    # Now S_i' = alpha^T (k_i * acc_vec_i)
    return (K * acc_vec * alpha[None, :]).sum(axis=1)

'''
JAX version
'''
@jax.jit
def _weighted_values_gl_core(K, alpha, x, w):
    """
    Core JAX kernel.
    K: (d,m)        alpha: (m,)
    x: (m_q,) nodes in [0,1]   w: (m_q,) weights
    returns: (d,)
    """
    B = 1.0 + x[:, None, None] * K[None, :, :]      # (m_q,d,m)

    
    # exclusive prefix
    pref = lax.cumprod(B, axis=1)
    pref = jnp.concatenate(
        [jnp.ones((B.shape[0], 1, B.shape[2]), dtype=B.dtype), pref[:, :-1, :]],
        axis=1
    )
    # exclusive suffix
    suf = lax.cumprod(B[:, ::-1, :], axis=1)[:, ::-1, :]
    suf = jnp.concatenate(
        [suf[:, 1:, :], jnp.ones((B.shape[0], 1, B.shape[2]), dtype=B.dtype)],
        axis=1
    )

    Q = pref * suf                                  # (m_q,d,m)
    acc = (w[:, None, None] * Q).sum(axis=0)        # (d,m)
    return (K * acc * alpha[None, :]).sum(axis=1)   # (d,)

def weighted_values_gl_jax_auto(K: np.ndarray, alpha: np.ndarray, m_q=None):
    """
    One-shot function: builds Gauss–Legendre nodes/weights (mapped to [0,1])
    and returns the weighted values.

    K: (d,m) numpy array
    alpha: (m,) numpy array
    m_q: number of Gauss–Legendre nodes
    returns: (d,) numpy array
    """

    if m_q == None:
        m_q = (K.shape[0] + 1) // 2   # ceil(d/2)
    
    # 1) Build nodes/weights on [-1,1], then map to [0,1]
    x_np, w_np = np.polynomial.legendre.leggauss(m_q)
    x_np = 0.5 * (x_np + 1.0)   # to [0,1]
    w_np = 0.5 * w_np

    # 2) Respect input dtype/device preference
    dtype = np.result_type(K.dtype, alpha.dtype, np.float32)
    x = jnp.asarray(x_np, dtype=dtype)
    w = jnp.asarray(w_np, dtype=dtype)
    Kj = jnp.asarray(K, dtype=dtype)
    alphaj = jnp.asarray(alpha, dtype=dtype)

    # 3) Call the jitted core
    out = _weighted_values_gl_core(Kj, alphaj, x, w)

    # 4) Return to numpy (or keep as JAX array if you prefer)
    return np.asarray(out)


def unweighted_values_from_kernel_vectors(kernel_vectors, alpha):
    """
    kernel_vectors: array of shape (d, m), each row is k_j (vector for feature j)
    alpha: array of shape (m,)
    Returns: weighted_value per feature, shape (d,)
    """
    K = np.asarray(kernel_vectors)       # shape (d, m)
    d, m = K.shape
    b = 1.0 + K                          # (d, m)
    
    # Compute global product across features (axis=0) → shape (m,)
    P = b.prod(axis=0)
    
    # Exclusive products: remove each feature
    # Avoid division by zero → use prefix/suffix products
    pref = np.cumprod(b, axis=0)         # prefix inclusive
    pref = np.vstack([np.ones((1, m)), pref[:-1]])   # make exclusive
    suf = np.cumprod(b[::-1], axis=0)[::-1]          # suffix inclusive reversed
    suf = np.vstack([suf[1:], np.ones((1, m))])      # make exclusive
    
    E = pref * suf   # E[i,:] = product of all b[j,:], j != i
    
    # Finally: S[i] = alpha dot (a_i * E[i])
    result = (K * E * alpha[None, :]).sum(axis=1)
    return result


@njit(fastmath=True, parallel=True)
def weighted_values_numba(K, mu, alpha):
    d, m = K.shape
    E = np.zeros((d+1, m), dtype=K.dtype)
    E[0, :] = 1.0

    # Step 1: global ESPs (parallelize across m inside each update)
    for j in range(d):
        for k in range(j+1, 0, -1):
            # E[k,:] += K[j,:] * E[k-1,:]
            # expand to loops so Numba can parallelize over m
            for t in prange(m):
                E[k, t] += K[j, t] * E[k-1, t]

    # Step 2: synthetic division for all j in parallel over (j,t)
    q_prev = np.ones((d, m), dtype=K.dtype)
    result = np.empty((d, m), dtype=K.dtype)

    # result = mu[0] * q_prev
    for j in prange(d):
        for t in range(m):
            result[j, t] = mu[0] * q_prev[j, t]

    for q in range(1, d):
        # q_curr = E[q] - K * q_prev
        for j in prange(d):
            for t in range(m):
                val = E[q, t] - K[j, t] * q_prev[j, t]
                result[j, t] += mu[q] * val
                q_prev[j, t] = val

    # Step 3: weighted dot with alpha
    weighted = np.zeros(d, dtype=K.dtype)
    for j in prange(d):
        s = 0.0
        for t in range(m):
            s += (K[j, t] - 1.0) * result[j, t] * alpha[t]
        weighted[j] = s
    return weighted

def weighted_values_from_kernel_vectors(kernel_vectors, mu_coefficients, alpha, E2):
    """
    kernel_vectors: list/array of length d with vectors of shape (m,)
                    or array of shape (d, m) where d=#features, m=vector length
    mu_coefficients: array of shape (d,) containing mu[q] for q=0..d-1
    alpha: array of shape (m,)
    Returns: weighted_value per feature, shape (d,)
    """
    K = np.asarray(kernel_vectors)               # shape (d, m)
    d, m = K.shape
    mu = np.asarray(mu_coefficients)             # shape (d,)
    alpha = np.asarray(alpha)                    # shape (m,)
    onevec = np.ones(m, dtype=K.dtype)

    # Step 1: global ESPs E[0..d], each E[k] shape (m,)
    E = np.zeros((d+1, m), dtype=K.dtype)
    E[0] = 1.0
    # classic in-place ESP recurrence, element-wise across m
    for j in range(d):
        # descending k to avoid overwrite hazards
        # E[k] += K[j] * E[k-1]
        # vectorized over m; loop over k only (d^2*m total ops)
        for k in range(j+1, 0, -1):
            E[k] += K[j] * E[k-1]

    # Step 2: synthetic division for ALL j in parallel
    # q_prev[j,:] holds q^{(j)}_{k-1} across all j
    q_prev = np.ones((d, m), dtype=K.dtype)      # q^{(j)}_0 = 1
    # result[j,:] accumulates sum_q mu[q] * q^{(j)}_q
    result = (mu[0] * q_prev)                    # add mu[0]*q0
    
    if E2 is not None:
        E = E2

    # Iterate q = 1..d-1; broadcast E[q] over j, K over (j,:)
    for q in range(1, d):
        q_curr = E[q][None, :] - K * q_prev      # shape (d, m)
        result += mu[q] * q_curr
        q_prev = q_curr

    # Step 3: per-feature weighted value = alpha dot ((k_j - 1) * result_j)
    # element-wise multiply over m, then reduce with alpha
    weighted_values = ((K - 1.0) * result * alpha[None, :]).sum(axis=1)  # shape (d,)
    return E, weighted_values

# ---------- Main Explainer Classes ----------

class ProductKernelLocalExplainer:
    def __init__(self, model):
        """
        Initialize the Shapley Value Explainer.

        Args:
            model: A scikit-learn model (GP, SVM or SVR) with RBF kernel
        """
        self.model = model
        self.X_train = self.get_X_train()
        self.alpha = self.get_alpha()
        self.n, self.d = self.X_train.shape

    def get_X_train(self):
        """
        Retrieve the training sample  based on the model type.

        Returns:
            2D-array of samples.
        """
        if hasattr(self.model, "support_vectors_"):  # For SVM/SVR
            return self.model.support_vectors_

        if hasattr(self.model, "X_fit_"):  # For KRR
            return self.model.X_fit_
        
        elif hasattr(self.model, "X_train_"):  # For GP
            return self.model.X_train_

        if hasattr(self.model, "base_estimator_") and hasattr(self.model.base_estimator_, "X_train_"):  # for GP classifier
            return self.model.base_estimator_.X_train_

        else:
            raise ValueError("Unsupported model type for Shapley value computation.")
        
    def get_alpha(self):
        """
        Retrieve the alpha values based on the model type.

        Returns:
            Array of alpha values required for Shapley value computation.
        """
        if hasattr(self.model, "dual_coef_"):  # For SVM/SVR
            self.null_game = self.model.intercept_
        
            return self.model.dual_coef_.flatten()
        
        elif hasattr(self.model, "alpha_"):  # For GP
            alpha = self.model.alpha_.flatten()
            self.null_game = np.sum(alpha)
            
            return alpha
        
        else:
            raise ValueError("Unsupported model type for Shapley value computation.")
    
    def precompute_mu(self, d):
        """
        Precompute mu coefficients (as in the paper) or the weights in Shapley values.

        Args:
            d: Number of features.

        Returns:
            List of precomputed mu coefficients.
        """

        unnormalized_factors = [(math.factorial(q) * math.factorial(d - q - 1)) for q in range(d)]

        return np.array(unnormalized_factors) / math.factorial(d) 
    
    def compute_elementary_symmetric_polynomials_recursive(self, kernel_vectors):
        """
        Compute elementary symmetric polynomials.

        Args:
            kernel_vectors: List of kernel vectors (1D arrays) of features 
                (for local explainer, it is computed by realizing kernel function between each feature of x (instance under explanation) and training set).

        Returns:
            e: List of elementary symmetric polynomials .
        """
    
        # Compute power sums
        s = [
            sum([np.power(k, p) for k in kernel_vectors])
            for p in range(0, len(kernel_vectors) + 1)
        ]
        
        # Compute elementary symmetric polynomials
        e = [np.ones_like(kernel_vectors[0])]  # e_0 = 1
        
        for r in range(1, len(kernel_vectors) + 1):
            term = 0 
            for k in range(1, r + 1):
                term += ((-1) ** (k-1)) * e[r - k] * s[k]
            e.append(term / r )
        
        return e

    def compute_elementary_symmetric_polynomials(self, kernel_vectors):
        """
        Compute elementary symmetric polynomials using a dynamic programming approach.

        Args:
            kernel_vectors: List of kernel vectors (1D arrays).

        Returns:
            elementary: List of elementary symmetric polynomials.
        """
        # Initialize with e_0 = 1
        max_abs_k = max(np.max(np.abs(k)) for k in kernel_vectors) or 1.0
        scaled_kernel = [k / max_abs_k for k in kernel_vectors]

        # Initialize polynomial coefficients: P_0(x) = 1
        e = [np.ones_like(scaled_kernel[0], dtype=np.float64)]

        for k in scaled_kernel:
            # Prepend and append zeros to handle polynomial multiplication (x - k)
            new_e = [np.zeros_like(e[0])] * (len(e) + 1)
            # new_e[0] corresponds to the constant term after multiplying by (x - k)
            new_e[0] = -k * e[0]
            # Compute the rest of the terms
            for i in range(1, len(e)):
                new_e[i] = e[i-1] - k * e[i]
            # The highest degree term is x^{len(e)}, coefficient is e[-1] (which is 1 initially)
            new_e[len(e)] = e[-1].copy()
            e = new_e
        
        # Extract elementary symmetric polynomials from the coefficients
        n = len(scaled_kernel)
        elementary = [np.ones_like(e[0])]  # e_0 = 1
        for r in range(1, n + 1):
            sign = (-1) ** r
            elementary_r = sign * e[n - r] * (max_abs_k ** r)
            elementary.append(elementary_r)
        
        return elementary
    
    def explain_by_kernel_vectors(self, kernel_vectors):
        """
        Compute Shapley values for all features of an instance based on computed feature-wise kernel vectors

        Args:
            kernel_vectors: feature-wise kernel vectors between x and training samples

        Returns:
            List of Shapley values, one for each feature.
        """

        shapley_values = []
        for j in range(self.d):
            shapley_values.append(self._compute_shapley_value(kernel_vectors, j))
        
        return shapley_values

    def _compute_shapley_value(self, kernel_vectors, feature_index):
        """
        Compute the Shapley value for a specific feature of an instance.

        Args:
            X: The dataset (2D array) used to train the model.
            x: The instance (1D array) for which to compute the Shapley value.
            feature_index: Index of the feature.

        Returns:
            Shapley value for the specified feature.
        """

        cZ_minus_j = [kernel_vectors[i] for i in range(self.d) if i != feature_index]
        e_polynomials = self.compute_elementary_symmetric_polynomials(cZ_minus_j)
        mu_coefficients = self.precompute_mu(self.d)
        
        # Compute kernel vector for the chosen feature
        k_j = kernel_vectors[feature_index]
        onevec = np.ones_like(k_j)
        
        # Compute the Shapley value
        result = np.zeros_like(k_j)
        for q in range(self.d):
            if q < len(e_polynomials):
                result += mu_coefficients[q] * e_polynomials[q]
        
        shapley_value = self.alpha.dot((k_j - onevec) * result)
        return shapley_value.item()
    
    def v_S(self, kernel_vectors, S):
        """
        Compute v(S): the inner product of alpha with the elementwise product of kernel_vectors columns in S.

        Args:
            kernel_vectors: list or np.ndarray of shape (d, n) or (n, d), kernel values for each feature and training point.
            S: iterable of indices (features to include).

        Returns:
            Scalar value: alpha^T (elementwise product of columns in S).
        """
        # Ensure kernel_vectors is (n, d)
        if isinstance(kernel_vectors, list):
            kernel_vectors = np.array(kernel_vectors).T  # shape (n, d)
        elif kernel_vectors.shape[0] != self.n:
            kernel_vectors = kernel_vectors.T  # shape (n, d)

        if len(S) == 0:
            prod = np.ones(self.n)
        else:
            prod = np.prod(kernel_vectors[:, list(S)], axis=1)
        return np.dot(self.alpha, prod)

    def brute_force_shapley(self, kernel_vectors):
        """
        Brute-force computation of Shapley values for all features using the Mobius representation.

        Args:
            kernel_vectors: np.ndarray of shape (n, d), kernel values for each training point and feature.
            alpha: np.ndarray of shape (n,), model coefficients.

        Returns:
            np.ndarray of Shapley values for all features (shape: d,).
        """
        import itertools
        n, d = kernel_vectors.shape
        shapley_values = np.zeros(d)

        features = list(range(d))
        # Iterate over all subsets S of features
        for subset_size in range(1, d + 1):
            for S in itertools.combinations(features, subset_size):
                # Compute m(S)
                prod_S = np.ones(n)
                for idx in S:
                    prod_S *= (kernel_vectors[:, idx] - 1)
                m_S = np.dot(self.alpha, prod_S)
                # Add contribution to all phi_i for i in S
                for i in S:
                    shapley_values[i] += (1.0 / subset_size) * m_S

        return shapley_values
        
    def explain_by_kernel_vectors_chebyshev(self, kernel_vectors):
        """
        Compute Shapley values for all features of an instance using
        a Chebyshev + prefix/suffix product representation instead of
        explicit elementary symmetric polynomial recursion.

        Args:
            kernel_vectors: list of length d, each element a 1D np.array of
                            shape (n,) with kernel values between the instance
                            x (under explanation) and the training points for
                            that feature.

        Returns:
            np.ndarray of shape (d,) with the Shapley value for each feature.
        """
        import numpy as np

        # ---- 0. Normalize / check shapes ---------------------------------
        # kernel_vectors can be a list of arrays or an array of shape (d, n) or (n, d)
        if isinstance(kernel_vectors, list):
            kv = np.stack(kernel_vectors, axis=0)  # shape (d, n)
        else:
            kv = np.asarray(kernel_vectors)
            if kv.ndim != 2:
                raise ValueError("kernel_vectors must be 2D (d, n) or (n, d).")
            # Ensure shape (d, n)
            if kv.shape[0] == self.n and kv.shape[1] == self.d:
                kv = kv.T
            elif kv.shape[0] != self.d or kv.shape[1] != self.n:
                raise ValueError(
                    f"kernel_vectors shape mismatch: expected (d={self.d}, n={self.n}) "
                    f"or (n={self.n}, d={self.d}), got {kv.shape}."
                )

        d = self.d
        n = self.n

        # Trivial 1D case: only one feature
        if d == 1:
            k0 = kv[0]
            onevec = np.ones_like(k0)
            # v({0}) - v(∅) ~ alpha^T (k0 - 1)
            return np.array([self.alpha.dot(k0 - onevec).item()])

        # ---- 1. Chebyshev nodes and Vandermonde --------------------------
        # Degree of H_{-j}(y) is d-1
        D = d - 1

        # Chebyshev points of the second kind on [-1, 1]
        y = np.cos(np.pi * np.arange(D + 1) / D)  # shape (D+1,)

        # Vandermonde matrix V_{ℓk} = y_ℓ^k, ℓ,k = 0..D
        # We will solve V^T N = c
        powers = np.arange(D + 1, dtype=float)
        V = y[:, None] ** powers[None, :]  # shape (D+1, D+1)

        # ---- 2. Coefficient-space weights c_k = μ_k ----------------------
        # μ_q are exactly the Shapley (reciprocal-binomial) weights in q
        mu = self.precompute_mu(d)  # length d
        c = mu[:D + 1]              # we only need 0..D

        # Solve V^T N = c   =>   (D+1)x(D+1) system
        # This gives N such that for any polynomial G with coeffs e_k,
        # N^T (V e) = c^T e = Σ_k μ_k e_k.
        N = np.linalg.solve(V.T.astype(float), c.astype(float))  # shape (D+1,)

        # ---- 3. Build factor evaluations F_j(ℓ, :) = 1 + k_j * y_ℓ ------
        # kv: (d, n), y: (D+1,)
        # We want F of shape (D+1, d, n)
        y_grid = y[:, None, None]         # (D+1, 1, 1)
        kv_grid = kv[None, :, :]          # (1, d, n)
        F = 1.0 + kv_grid * y_grid        # (D+1, d, n)

        # ---- 4. Prefix and suffix products in Chebyshev space -----------
        # P[k] = product over features 0..k-1 of F[:, j, :]
        # S[k] = product over features k..d-1 of F[:, j, :]
        # We store P, S as (D+1, d+1, n)
        P = np.ones((D + 1, d + 1, n), dtype=np.float64)
        for j in range(d):
            # P[:, j+1, :] = P[:, j, :] * F[:, j, :]
            P[:, j + 1, :] = P[:, j, :] * F[:, j, :]

        S = np.ones((D + 1, d + 1, n), dtype=np.float64)
        for j in reversed(range(d)):
            # S[:, j, :] = S[:, j+1, :] * F[:, j, :]
            S[:, j, :] = S[:, j + 1, :] * F[:, j, :]

        # ---- 5. For each feature i, assemble H_{-i} and apply functional ψ ----
        shapley_values = np.zeros(d, dtype=float)
        onevec = np.ones(n, dtype=float)

        for i in range(d):
            # H_{-i}(y_ℓ) = Π_{j≠i} (1 + k_j y_ℓ) = P[:, i, :] * S[:, i+1, :]
            H_minus_i = P[:, i, :] * S[:, i + 1, :]    # shape (D+1, n)

            # result(q-aggregated ESP combination) at each training sample:
            # result = Σ_q μ_q e_q  =  N^T H_minus_i (in Chebyshev eval space)
            # N: (D+1,), H_minus_i: (D+1, n) → result: (n,)
            result = N @ H_minus_i   # matmul over ℓ

            # Final Shapley value for feature i:
            # φ_i = alpha^T [ (k_i - 1) ⊙ result ]
            k_i = kv[i]              # shape (n,)
            shapley_values[i] = self.alpha.dot((k_i - onevec) * result).item()

        return shapley_values

    def explain_by_kernel_vectors_chebyshev_log(self, kernel_vectors):
        """
        Compute Shapley values for all features of an instance using
        a Chebyshev + prefix/suffix product representation, but
        performing the prefix/suffix products in log-space for
        numerical stability.

        This version keeps the Chebyshev interpolation and the
        Vandermonde system exactly as in the original method, but
        replaces direct products over features by log-domain
        prefix/suffix accumulation with sign tracking.
        """
        import numpy as np

        # ---- 0. Normalize / check shapes ---------------------------------
        if isinstance(kernel_vectors, list):
            kv = np.stack(kernel_vectors, axis=0)  # shape (d, n)
        else:
            kv = np.asarray(kernel_vectors)
            if kv.ndim != 2:
                raise ValueError("kernel_vectors must be 2D (d, n) or (n, d).")
            # Ensure shape (d, n)
            if kv.shape[0] == self.n and kv.shape[1] == self.d:
                kv = kv.T
            elif kv.shape[0] != self.d or kv.shape[1] != self.n:
                raise ValueError(
                    f"kernel_vectors shape mismatch: expected (d={self.d}, n={self.n}) "
                    f"or (n={self.n}, d={self.n}), got {kv.shape}."
                )

        d = self.d
        n = self.n

        # Trivial 1D case: only one feature
        if d == 1:
            k0 = kv[0]
            onevec = np.ones_like(k0)
            return np.array([self.alpha.dot(k0 - onevec).item()])

        # ---- 1. Chebyshev nodes and Vandermonde --------------------------
        D = d - 1  # degree of H_{-j}

        # Chebyshev points of the second kind on [-1, 1]
        y = np.cos(np.pi * np.arange(D + 1) / D)  # shape (D+1,)

        # Vandermonde matrix V_{ℓk} = y_ℓ^k, ℓ,k = 0..D
        powers = np.arange(D + 1, dtype=float)
        V = y[:, None] ** powers[None, :]  # (D+1, D+1)

        # ---- 2. Coefficient-space weights c_k = μ_k ----------------------
        mu = self.precompute_mu(d)   # length d
        c = mu[:D + 1]               # use 0..D

        # Solve V^T N = c for Chebyshev weights N
        N = np.linalg.solve(V.T.astype(float), c.astype(float))  # (D+1,)

        # ---- 3. Build factor evaluations F[ℓ, j, s] = 1 + k_j(s) * y_ℓ ---
        y_grid = y[:, None, None]         # (D+1, 1, 1)
        kv_grid = kv[None, :, :]          # (1, d, n)
        F = 1.0 + kv_grid * y_grid        # (D+1, d, n), real

        # ---- 4. Log-space prefix & suffix products over features ---------
        # We represent products in the form:
        #   product = sign * exp(log_amp)
        # where log_amp = sum(log |F|), sign = product(sign(F)).
        #
        # To avoid log(0), we add a tiny epsilon to the magnitude.
        eps = 1e-300
        absF = np.abs(F) + eps
        logF = np.log(absF)               # (D+1, d, n)
        signF = np.sign(F)                # (D+1, d, n), in {-1, 0, +1}

        # Prefix: logP_amp[:, j, :] = sum_{t<j} logF[:, t, :]
        #         signP[:, j, :]    = product_{t<j} signF[:, t, :]
        logP_amp = np.zeros((D + 1, d + 1, n), dtype=np.float64)
        signP = np.ones((D + 1, d + 1, n), dtype=np.float64)

        for j in range(d):
            logP_amp[:, j + 1, :] = logP_amp[:, j, :] + logF[:, j, :]
            signP[:, j + 1, :] = signP[:, j, :] * signF[:, j, :]

        # Suffix: logS_amp[:, j, :] = sum_{t>=j} logF[:, t, :]
        #         signS[:, j, :]    = product_{t>=j} signF[:, t, :]
        logS_amp = np.zeros((D + 1, d + 1, n), dtype=np.float64)
        signS = np.ones((D + 1, d + 1, n), dtype=np.float64)

        for j in reversed(range(d)):
            logS_amp[:, j, :] = logS_amp[:, j + 1, :] + logF[:, j, :]
            signS[:, j, :] = signS[:, j + 1, :] * signF[:, j, :]

        # ---- 5. For each feature i, reconstruct H_{-i} from log+sign -----
        shapley_values = np.zeros(d, dtype=float)
        onevec = np.ones(n, dtype=float)

        for i in range(d):
            # log H_{-i}(y_ℓ,:) = logP_amp[:, i, :] + logS_amp[:, i+1, :]
            # sign H_{-i}(y_ℓ,:) = signP[:, i, :] * signS[:, i+1, :]
            logH = logP_amp[:, i, :] + logS_amp[:, i + 1, :]     # (D+1, n)
            signH = signP[:, i, :] * signS[:, i + 1, :]          # (D+1, n)

            # Stability trick: for each sample s, subtract max log over ℓ
            # so that exp(logH_shifted) has real part ≤ 0.
            # base: (n,) with base[s] = max_ℓ logH[ℓ,s]
            base = np.max(logH, axis=0)                          # (n,)

            # Shift logs: logH_shifted[ℓ,s] = logH[ℓ,s] - base[s]
            logH_shifted = logH - base[None, :]                  # (D+1, n)

            # Reconstruct scaled H_{-i}(y_ℓ,:):
            #   H_{-i} = signH * exp(logH) = exp(base) * (signH * exp(logH_shifted))
            H_scaled = signH * np.exp(logH_shifted)              # (D+1, n)

            # Apply Chebyshev weights N: result_s = Σ_ℓ N_ℓ H_{-i}(y_ℓ,s)
            # Then multiply by exp(base[s]).
            tmp = N @ H_scaled                                   # (n,)
            scale = np.exp(base)                                 # (n,)
            result = (scale * tmp).astype(np.float64)            # (n,)

            # Final Shapley value for feature i:
            # φ_i = alpha^T [ (k_i - 1) ⊙ result ]
            k_i = kv[i]                                          # (n,)
            shapley_values[i] = self.alpha.dot((k_i - onevec) * result).item()

        return shapley_values

    def explain_by_kernel_vectors_fft(self, kernel_vectors):
        """
        FFT-based version of explain_by_kernel_vectors_chebyshev:
        uses roots of unity + FFT instead of Chebyshev nodes + Vandermonde inverse.

        Args:
            kernel_vectors: list of length d, each element a 1D np.array of
                            shape (n,) with kernel values between the instance
                            x (under explanation) and the training points for
                            that feature.

        Returns:
            np.ndarray of shape (d,) with the Shapley value for each feature.
        """
        import numpy as np

        # ---- 0. Normalize / check shapes ---------------------------------
        if isinstance(kernel_vectors, list):
            kv = np.stack(kernel_vectors, axis=0)  # shape (d, n)
        else:
            kv = np.asarray(kernel_vectors)
            if kv.ndim != 2:
                raise ValueError("kernel_vectors must be 2D (d, n) or (n, d).")
            # Ensure shape (d, n)
            if kv.shape[0] == self.n and kv.shape[1] == self.d:
                kv = kv.T
            elif kv.shape[0] != self.d or kv.shape[1] != self.n:
                raise ValueError(
                    f"kernel_vectors shape mismatch: expected (d={self.d}, n={self.n}) "
                    f"or (n={self.n}, d={self.d}), got {kv.shape}."
                )

        d = self.d
        n = self.n

        # Trivial 1D case: only one feature
        if d == 1:
            k0 = kv[0]
            onevec = np.ones_like(k0)
            return np.array([self.alpha.dot(k0 - onevec).item()])

        # ------------------------------------------------------------------
        # 1. Roots of unity & FFT-based weights instead of Chebyshev/Vandermonde
        # ------------------------------------------------------------------
        # Degree of H_{-j}(z) ≤ d-1
        D = d - 1
        n_fft = D + 1  # minimal length so Vandermonde on roots of unity is invertible

        # n_fft-th roots of unity ω_ℓ = exp(2π i ℓ / n_fft)
        ell = np.arange(n_fft, dtype=float)
        omega = np.exp(2j * np.pi * ell / n_fft)  # shape (n_fft,), complex

        # μ_q are the Shapley weights over degrees q
        mu = self.precompute_mu(d)  # length d, real

        # Pad μ to length n_fft (higher degrees have coefficient 0)
        c = np.zeros(n_fft, dtype=np.complex128)
        c[:D + 1] = mu[:D + 1].astype(np.complex128)

        # FFT-based linear functional N such that for any polynomial G(z)
        # with coefficients e_q, we have:
        #     Σ_q μ_q e_q  =  Σ_ℓ N_ℓ G(ω_ℓ)
        #
        # Using Vandermonde on roots of unity, one gets:
        #     N = (1/n_fft) * FFT(μ-padded)
        N = np.fft.fft(c) / n_fft   # shape (n_fft,), complex

        # ------------------------------------------------------------------
        # 2. Build factor evaluations F[ℓ, j, s] = 1 + k_j(s) * ω_ℓ
        # ------------------------------------------------------------------
        # kv: (d, n), omega: (n_fft,)
        omega_grid = omega[:, None, None]   # (n_fft, 1, 1)
        kv_grid = kv[None, :, :]            # (1, d, n)
        F = 1.0 + kv_grid * omega_grid      # (n_fft, d, n), complex

        # ------------------------------------------------------------------
        # 3. Prefix and suffix products in Fourier space
        # ------------------------------------------------------------------
        # P[:, k, :] = ∏_{j=0}^{k-1} F[:, j, :]
        # S[:, k, :] = ∏_{j=k}^{d-1} F[:, j, :]
        P = np.ones((n_fft, d + 1, n), dtype=np.complex128)
        for j in range(d):
            P[:, j + 1, :] = P[:, j, :] * F[:, j, :]

        S = np.ones((n_fft, d + 1, n), dtype=np.complex128)
        for j in reversed(range(d)):
            S[:, j, :] = S[:, j + 1, :] * F[:, j, :]

        # ------------------------------------------------------------------
        # 4. For each feature i, assemble H_{-i}(ω_ℓ) and apply functional
        # ------------------------------------------------------------------
        shapley_values = np.zeros(d, dtype=float)
        onevec = np.ones(n, dtype=float)

        for i in range(d):
            # H_{-i}(ω_ℓ, s) = ∏_{j≠i} (1 + k_j(s) ω_ℓ)
            # via prefix/suffix:
            H_minus_i = P[:, i, :] * S[:, i + 1, :]  # (n_fft, n), complex

            # For each sample s, result_s = Σ_q μ_q e_q^{(-i)}(s)
            # = Σ_ℓ N_ℓ H_{-i}(ω_ℓ, s)
            result_complex = N @ H_minus_i          # (n,)
            result = result_complex.real.astype(np.float64)

            # Final Shapley value for feature i:
            # φ_i = α^T [ (k_i - 1) ⊙ result ]
            k_i = kv[i]  # (n,)
            shapley_values[i] = self.alpha.dot((k_i - onevec) * result).item()

        return shapley_values
    
    def explain_by_kernel_vectors_fft_scaled(self, kernel_vectors):
        """
        FFT-based Shapley calculator with per-feature scaling to improve
        numerical stability when kernel values k_j can be > 1.

        - Uses roots of unity + FFT for the linear functional over degrees.
        - Replaces Chebyshev + Vandermonde.
        - Multiplies bounded complex factors G_j (|G_j| <= 1) via prefix/suffix,
        and tracks a separate real scale per (i, sample).

        Args:
            kernel_vectors: list of length d, each element a 1D np.array of
                            shape (n,) with kernel values between the instance
                            x (under explanation) and the training points for
                            that feature.

        Returns:
            np.ndarray of shape (d,) with the Shapley value for each feature.
        """
        import numpy as np

        # ---- 0. Normalize / check shapes ---------------------------------
        if isinstance(kernel_vectors, list):
            kv = np.stack(kernel_vectors, axis=0)  # shape (d, n)
        else:
            kv = np.asarray(kernel_vectors)
            if kv.ndim != 2:
                raise ValueError("kernel_vectors must be 2D (d, n) or (n, d).")
            # Ensure shape (d, n)
            if kv.shape[0] == self.n and kv.shape[1] == self.d:
                kv = kv.T
            elif kv.shape[0] != self.d or kv.shape[1] != self.n:
                raise ValueError(
                    f"kernel_vectors shape mismatch: expected (d={self.d}, n={self.n}) "
                    f"or (n={self.n}, d={self.d}), got {kv.shape}."
                )

        d = self.d
        n = self.n

        # Trivial 1D case: only one feature
        if d == 1:
            k0 = kv[0]
            onevec = np.ones_like(k0)
            return np.array([self.alpha.dot(k0 - onevec).item()])

        # ------------------------------------------------------------------
        # 1. Roots of unity & FFT-based weights for Shapley degree weights
        # ------------------------------------------------------------------
        D = d - 1
        n_fft = D + 1  # minimal length capturing all degrees 0..D

        ell = np.arange(n_fft, dtype=float)
        omega = np.exp(2j * np.pi * ell / n_fft)  # (n_fft,), complex

        mu = self.precompute_mu(d)  # length d, real
        c = np.zeros(n_fft, dtype=np.complex128)
        c[:D + 1] = mu[:D + 1].astype(np.complex128)

        # For any polynomial G with coeffs e_q, Σ_q μ_q e_q = Σ_ℓ N_ℓ G(ω_ℓ)
        N = np.fft.fft(c) / n_fft   # (n_fft,), complex

        # ------------------------------------------------------------------
        # 2. Build F_j(ℓ,s) = 1 + k_j(s) ω_ℓ and scaled G_j = F_j / b_j
        # ------------------------------------------------------------------
        # kv: (d, n), omega: (n_fft,)
        omega_grid = omega[:, None, None]         # (n_fft, 1, 1)
        kv_grid = kv[None, :, :]                  # (1, d, n)

        F = 1.0 + kv_grid * omega_grid            # (n_fft, d, n), complex

        # Per-feature, per-sample real scale: b_j,s = 1 + |k_j,s|
        # This bounds |F_j,ℓ,s| ≤ b_j,s for all ℓ.
        b = 1.0 + np.abs(kv)                      # (d, n), real > 0
        log_b = np.log(b)                         # (d, n), real

        b_grid = b[None, :, :]                    # (1, d, n)
        G = F / b_grid                            # (n_fft, d, n), complex
        # Now |G_j,ℓ,s| <= 1 for all j,ℓ,s.

        # ------------------------------------------------------------------
        # 3. Prefix and suffix products on G, plus log-scales of b
        # ------------------------------------------------------------------
        # P_val[:, k, :] = ∏_{j=0}^{k-1} G[:, j, :]
        # S_val[:, k, :] = ∏_{j=k}^{d-1} G[:, j, :]
        # P_log[k, s]    = Σ_{j=0}^{k-1} log b_{j,s}
        # S_log[k, s]    = Σ_{j=k}^{d-1} log b_{j,s}

        P_val = np.ones((n_fft, d + 1, n), dtype=np.complex128)
        P_log = np.zeros((d + 1, n), dtype=np.float64)

        for j in range(d):
            P_val[:, j + 1, :] = P_val[:, j, :] * G[:, j, :]
            P_log[j + 1, :] = P_log[j, :] + log_b[j, :]

        S_val = np.ones((n_fft, d + 1, n), dtype=np.complex128)
        S_log = np.zeros((d + 1, n), dtype=np.float64)

        for j in reversed(range(d)):
            S_val[:, j, :] = S_val[:, j + 1, :] * G[:, j, :]
            S_log[j, :] = S_log[j + 1, :] + log_b[j, :]

        # ------------------------------------------------------------------
        # 4. For each feature i, assemble H_{-i} and apply linear functional
        # ------------------------------------------------------------------
        shapley_values = np.zeros(d, dtype=float)
        onevec = np.ones(n, dtype=float)

        for i in range(d):
            # Complex product over G:
            #   G_prod(ℓ,s) = ∏_{j≠i} G_j,ℓ,s
            # via prefix/suffix:
            G_prod = P_val[:, i, :] * S_val[:, i + 1, :]  # (n_fft, n), complex

            # Real scale:
            #   log B_{i,s} = Σ_{j≠i} log b_{j,s}
            log_B = P_log[i, :] + S_log[i + 1, :]         # (n,), real

            # We keep B as a separate real factor:
            B = np.exp(log_B)                             # (n,), real

            # Now H_{-i}(ω_ℓ,s) = B_{i,s} * G_prod(ℓ,s)
            # So result_s = Σ_ℓ N_ℓ H_{-i}(ω_ℓ,s)
            #            = B_{i,s} * Σ_ℓ N_ℓ G_prod(ℓ,s)
            tmp = N @ G_prod                               # (n,), complex
            result_complex = B * tmp                      # (n,), complex
            result = result_complex.real.astype(np.float64)

            k_i = kv[i]                                   # (n,)
            shapley_values[i] = self.alpha.dot((k_i - onevec) * result).item()

        return shapley_values

    def explain_by_kernel_vectors_fft2(self, kernel_vectors):
        """
        FFT-based version of explain_by_kernel_vectors_chebyshev:
        uses roots of unity + FFT instead of Chebyshev nodes + Vandermonde inverse.

        Args:
            kernel_vectors: list of length d, each element a 1D np.array of
                            shape (n,) with kernel values between the instance
                            x (under explanation) and the training points for
                            that feature.

        Returns:
            np.ndarray of shape (d,) with the Shapley value for each feature.
        """
        import numpy as np

        # ---- 0. Normalize / check shapes ---------------------------------
        if isinstance(kernel_vectors, list):
            kv = np.stack(kernel_vectors, axis=0)  # shape (d, n)
        else:
            kv = np.asarray(kernel_vectors)
            if kv.ndim != 2:
                raise ValueError("kernel_vectors must be 2D (d, n) or (n, d).")
            # Ensure shape (d, n)
            if kv.shape[0] == self.n and kv.shape[1] == self.d:
                kv = kv.T
            elif kv.shape[0] != self.d or kv.shape[1] != self.n:
                raise ValueError(
                    f"kernel_vectors shape mismatch: expected (d={self.d}, n={self.n}) "
                    f"or (n={self.n}, d={self.d}), got {kv.shape}."
                )

        d = self.d
        n = self.n

        # Trivial 1D case: only one feature
        if d == 1:
            k0 = kv[0]
            onevec = np.ones_like(k0)
            return np.array([self.alpha.dot(k0 - onevec).item()])

        # ------------------------------------------------------------------
        # 1. Roots of unity & FFT-based weights instead of Chebyshev/Vandermonde
        # ------------------------------------------------------------------
        # Degree of H_{-j}(z) ≤ d-1
        D = d - 1
        n_fft = D + 1  # minimal length so Vandermonde on roots of unity is invertible

        # n_fft-th roots of unity ω_ℓ = exp(2π i ℓ / n_fft)
        ell = np.arange(n_fft, dtype=float)
        omega = np.exp(2j * np.pi * ell / n_fft)  # shape (n_fft,), complex

        # μ_q are the Shapley weights over degrees q
        mu = self.precompute_mu(d)  # length d, real

        # Pad μ to length n_fft (higher degrees have coefficient 0)
        c = np.zeros(n_fft, dtype=np.complex128)
        c[:D + 1] = mu[:D + 1].astype(np.complex128)

        # FFT-based linear functional N such that for any polynomial G(z)
        # with coefficients e_q, we have:
        #     Σ_q μ_q e_q  =  Σ_ℓ N_ℓ G(ω_ℓ)
        #
        # Using Vandermonde on roots of unity, one gets:
        #     N = (1/n_fft) * FFT(μ-padded)
        N = np.fft.fft(c) / n_fft   # shape (n_fft,), complex

        # ------------------------------------------------------------------
        # 2. Build factor evaluations F[ℓ, j, s] = 1 + k_j(s) * ω_ℓ
        # ------------------------------------------------------------------
        omega_grid = omega[:, None, None]   # (n_fft, 1, 1)
        kv_grid = kv[None, :, :]            # (1, d, n)
        F = 1.0 + kv_grid * omega_grid      # (n_fft, d, n), complex

        # ------------------------------------------------------------------
        # 3. Prefix and suffix products in Fourier space
        # ------------------------------------------------------------------
        # P[:, k, :] = ∏_{j=0}^{k-1} F[:, j, :]
        # S[:, k, :] = ∏_{j=k}^{d-1} F[:, j, :]

        P = np.ones((n_fft, d + 1, n), dtype=np.complex128)
        for j in range(d):
            P[:, j + 1, :] = P[:, j, :] * F[:, j, :]

        S = np.ones((n_fft, d + 1, n), dtype=np.complex128)
        for j in reversed(range(d)):
            S[:, j, :] = S[:, j + 1, :] * F[:, j, :]

        # ------------------------------------------------------------------
        # 4. For each feature i, assemble H_{-i}(ω) and apply functional
        # ------------------------------------------------------------------
        shapley_values = np.zeros(d, dtype=float)
        onevec = np.ones(n, dtype=float)

        for i in range(d):
            # H_{-i}(ω_ℓ, s) = ∏_{j≠i} (1 + k_j(s) ω_ℓ)
            # via prefix/suffix:
            H_minus_i = P[:, i, :] * S[:, i + 1, :]  # (n_fft, n), complex

            # For each sample s, result_s = Σ_q μ_q e_q^{(-i)}(s)
            # = Σ_ℓ N_ℓ H_{-i}(ω_ℓ, s)
            result_complex = N @ H_minus_i          # (n,)
            result = result_complex.real.astype(np.float64)  # should be real

            k_i = kv[i]  # (n,)
            shapley_values[i] = self.alpha.dot((k_i - onevec) * result).item()

        return shapley_values

    def explain_by_kernel_vectors_fft_log(self, kernel_vectors):
        """
        Fully numerically stable FFT-based Shapley calculator:
        - Uses roots of unity + FFT (stable)
        - AND log-space prefix/suffix multiplication (stabilizes feature-dimension growth)
        - Recenters real log terms per sample before exponentiating to
          avoid overflow/underflow for large d (e.g., 100 features).
        """
        import numpy as np

        # ---- Normalize shapes ----
        if isinstance(kernel_vectors, list):
            kv = np.stack(kernel_vectors, axis=0)
        else:
            kv = np.asarray(kernel_vectors)
            if kv.ndim != 2:
                raise ValueError("kernel_vectors must be 2D.")
            if kv.shape == (self.n, self.d):
                kv = kv.T
            if kv.shape != (self.d, self.n):
                raise ValueError("shape mismatch.")
        d, n = self.d, self.n

        # ---- 1D trivial case ----
        if d == 1:
            k0 = kv[0]
            return np.array([self.alpha.dot(k0 - 1).item()])

        # ================================================================
        # 1. Roots of unity & FFT weight vector N
        # ================================================================
        D = d - 1
        n_fft = D + 1

        ell = np.arange(n_fft)
        omega = np.exp(2j * np.pi * ell / n_fft)  # (n_fft,)

        mu = self.precompute_mu(d)
        c = np.zeros(n_fft, dtype=np.complex128)
        c[:D+1] = mu[:D+1]
        N = np.fft.fft(c) / n_fft  # FFT weight vector

        # ================================================================
        # 2. Compute F = 1 + k_j ω_ℓ in polar form: magnitude & angle
        # ================================================================
        # kv: (d,n), omega: (n_fft,)
        omega_grid = omega[:, None, None]
        kv_grid = kv[None, :, :]
        F = 1.0 + kv_grid * omega_grid      # (n_fft, d, n), complex

        # Work in log magnitude + unwrapped phase to avoid branch-cut jumps.
        log_abs = np.log(np.abs(F))                 # (n_fft, d, n)
        ang = np.unwrap(np.angle(F), axis=1)        # unwrap across features

        # Prefix/suffix sums of log magnitude and phase.
        logP = np.zeros((n_fft, d + 1, n))
        angP = np.zeros((n_fft, d + 1, n))
        for j in range(d):
            logP[:, j + 1, :] = logP[:, j, :] + log_abs[:, j, :]
            angP[:, j + 1, :] = angP[:, j, :] + ang[:, j, :]

        logS = np.zeros((n_fft, d + 1, n))
        angS = np.zeros((n_fft, d + 1, n))
        for j in reversed(range(d)):
            logS[:, j, :] = logS[:, j + 1, :] + log_abs[:, j, :]
            angS[:, j, :] = angS[:, j + 1, :] + ang[:, j, :]

        shap = np.zeros(d)
        one = np.ones(n)

        for i in range(d):
            # log H_{-i}(ω_ℓ, s) = log magnitude + i * phase
            logH_real = logP[:, i, :] + logS[:, i + 1, :]      # (n_fft, n)
            logH_imag = angP[:, i, :] + angS[:, i + 1, :]      # (n_fft, n)

            # Center real part per sample to avoid overflow; keep phase intact.
            base = logH_real.max(axis=0)                      # (n,)
            logH_real_centered = logH_real - base[None, :]    # (n_fft, n)

            # Reconstruct safely: exp(real) * exp(i*phase)
            H_tilde = np.exp(logH_real_centered) * np.exp(1j * logH_imag)
            tmp = N @ H_tilde                                 # (n,), complex

            # Rescale in higher precision, splitting to avoid exp overflow.
            base_lp = base.astype(np.longdouble)
            base_safe = np.clip(base_lp, None, 600.0)         # keep exp below overflow
            rem = base_lp - base_safe
            scale_safe = np.exp(base_safe)                    # (n,)
            scale_rem = np.exp(rem)                           # (n,)
            result = (scale_safe * scale_rem * tmp).real.astype(np.float64)

            k_i = kv[i]
            shap[i] = self.alpha.dot((k_i - one) * result).item()

        return shap
   
    def explain_by_kernel_vectors_fft_log2(self, kernel_vectors):
        """
        More stable FFT-based Shapley:
        - Uses roots of unity FFT (no Vandermonde)
        - Evaluates polynomial at r * ω_ℓ with 0<r<=1 to shrink magnitudes
        - Uses log-space prefix/suffix to stabilize products across features.
        """
        import numpy as np

        # ---- Normalize shapes ----
        if isinstance(kernel_vectors, list):
            kv = np.stack(kernel_vectors, axis=0)
        else:
            kv = np.asarray(kernel_vectors)
            if kv.ndim != 2:
                raise ValueError("kernel_vectors must be 2D.")
            if kv.shape == (self.n, self.d):
                kv = kv.T
            if kv.shape != (self.d, self.n):
                raise ValueError("shape mismatch.")
        d, n = self.d, self.n

        # ---- 1D trivial case ----
        if d == 1:
            k0 = kv[0]
            return np.array([self.alpha.dot(k0 - 1).item()])

        # ================================================================
        # 1. Roots of unity & scaled FFT weight vector N'
        # ================================================================
        D = d - 1
        n_fft = D + 1

        ell = np.arange(n_fft)
        omega = np.exp(2j * np.pi * ell / n_fft)  # (n_fft,)

        # Choose scaling r based on max |k|
        max_abs_k = float(np.max(np.abs(kv)))
        if max_abs_k == 0:
            r = 1.0
        else:
            r = min(1.0, 0.9 / (max_abs_k + 1e-8))

        mu = self.precompute_mu(d)  # length d

        # μ'_q = μ_q / r^q (for q=0..D)
        q_arr = np.arange(D + 1, dtype=float)
        r_powers = r ** q_arr
        mu_scaled = mu[:D+1] / r_powers

        c = np.zeros(n_fft, dtype=np.complex128)
        c[:D+1] = mu_scaled.astype(np.complex128)

        # N' = FFT(μ_scaled)/n_fft
        N = np.fft.fft(c) / n_fft  # (n_fft,)

        # ================================================================
        # 2. Compute F = 1 + r * k_j * ω_ℓ in polar form
        # ================================================================
        omega_grid = (r * omega)[:, None, None]  # scaled roots
        kv_grid = kv[None, :, :]                 # (1,d,n)
        F = 1.0 + kv_grid * omega_grid           # (n_fft, d, n), complex

        # Add small epsilon inside abs to avoid log(0)
        eps = 1e-16
        log_abs = np.log(np.abs(F) + eps)             # (n_fft, d, n)
        ang = np.unwrap(np.angle(F), axis=1)          # unwrap over features

        # Prefix sums
        logP = np.zeros((n_fft, d + 1, n))
        angP = np.zeros((n_fft, d + 1, n))
        for j in range(d):
            logP[:, j + 1, :] = logP[:, j, :] + log_abs[:, j, :]
            angP[:, j + 1, :] = angP[:, j, :] + ang[:, j, :]

        # Suffix sums
        logS = np.zeros((n_fft, d + 1, n))
        angS = np.zeros((n_fft, d + 1, n))
        for j in reversed(range(d)):
            logS[:, j, :] = logS[:, j + 1, :] + log_abs[:, j, :]
            angS[:, j, :] = angS[:, j + 1, :] + ang[:, j, :]

        shap = np.zeros(d)
        one = np.ones(n)

        for i in range(d):
            # log H_{-i}(r ω_ℓ, s)
            logH_real = logP[:, i, :] + logS[:, i + 1, :]   # (n_fft, n)
            logH_imag = angP[:, i, :] + angS[:, i + 1, :]   # (n_fft, n)

            # Center real part per sample to avoid overflow
            base = np.max(logH_real, axis=0)                # (n,)
            logH_centered = logH_real - base[None, :]       # (n_fft, n)

            # Reconstruct
            H_tilde = np.exp(logH_centered) * np.exp(1j * logH_imag)  # (n_fft, n)
            tmp = N @ H_tilde                                        # (n,), complex

            # Rescale; clip base to keep exp in double range
            base_clipped = np.clip(base, -700.0, 700.0)
            scale = np.exp(base_clipped)                              # (n,)
            result = (scale * tmp).real.astype(np.float64)

            k_i = kv[i]
            shap[i] = self.alpha.dot((k_i - one) * result).item()

        return shap

    def explain_by_kernel_vectors_fft_log_direct(self, kernel_vectors):
        """
        Numerically stable FFT-based Shapley computation for product-kernel models,
        WITHOUT prefix/suffix products and WITHOUT radius scaling.

        - Uses roots of unity + FFT (no Vandermonde inversion).
        - Evaluates H_{-i}(t) at t = ω_ℓ (roots of unity), same as
        explain_by_kernel_vectors_fft.
        - Uses log-space (log-magnitude + unwrapped phase) so we never form
        huge/small products explicitly.
        - For each feature i, obtains H_{-i}(ω_ℓ) by subtracting its log-
        contribution from the total log-sum (no prefix/suffix).

        This is mathematically equivalent to explain_by_kernel_vectors_fft
        up to floating-point roundoff.

        Args
        ----
        kernel_vectors : list of length d or np.ndarray
            Either a list of length d, with each element a 1D array of length n
            (kernel values k_i(X_i, x_i)), or a 2D array of shape (d, n) or (n, d).

        Returns
        -------
        shapley_values : np.ndarray, shape (d,)
            Shapley value for each feature.
        """
        import numpy as np

        # ---- Normalize shapes: kv -> (d, n) ---------------------------------
        if isinstance(kernel_vectors, list):
            kv = np.stack(kernel_vectors, axis=0)
        else:
            kv = np.asarray(kernel_vectors)
            if kv.ndim != 2:
                raise ValueError("kernel_vectors must be 2D (d, n) or (n, d).")
            if kv.shape == (self.n, self.d):
                kv = kv.T
            if kv.shape != (self.d, self.n):
                raise ValueError(
                    f"kernel_vectors shape mismatch: expected (d={self.d}, n={self.n}) "
                    f"or (n={self.n}, d={self.d}), got {kv.shape}."
                )

        d, n = self.d, self.n

        # ---- Trivial 1D case -------------------------------------------------
        if d == 1:
            k0 = kv[0]
            onevec = np.ones_like(k0)
            return np.array([self.alpha.dot(k0 - onevec).item()])

        # ============================================================
        # 1. Roots of unity & FFT weights (same as explain_by_kernel_vectors_fft)
        # ============================================================
        D = d - 1
        n_fft = D + 1

        ell = np.arange(n_fft, dtype=float)
        omega = np.exp(2j * np.pi * ell / n_fft)  # (n_fft,), complex roots of unity

        # Shapley weights μ_q, q = 0..d-1
        mu = self.precompute_mu(d)  # shape (d,)

        # Pad μ to length n_fft and compute N = FFT(μ_padded)/n_fft
        c = np.zeros(n_fft, dtype=np.complex128)
        c[:D + 1] = mu[:D + 1].astype(np.complex128)
        N = np.fft.fft(c) / n_fft  # (n_fft,), complex

        # ============================================================
        # 2. Compute F = 1 + k_j * ω_ℓ and its log-magnitude/phase
        # ============================================================
        # kv: (d, n), omega: (n_fft,)
        omega_grid = omega[:, None, None]  # (n_fft, 1, 1)
        kv_grid = kv[None, :, :]           # (1, d, n)
        F = 1.0 + kv_grid * omega_grid     # (n_fft, d, n), complex

        # Log-magnitude (add small eps to avoid log(0)) and unwrapped phase
        eps = 1e-16
        log_abs = np.log(np.abs(F) + eps)        # (n_fft, d, n)
        ang = np.unwrap(np.angle(F), axis=1)     # unwrap over features j

        # Sum over all features once (NO prefix/suffix)
        # total_log_abs[ℓ, s] = Σ_j log|1 + k_j(s) ω_ℓ|
        # total_ang[ℓ, s]     = Σ_j arg(1 + k_j(s) ω_ℓ)
        total_log_abs = np.sum(log_abs, axis=1)  # (n_fft, n)
        total_ang = np.sum(ang, axis=1)          # (n_fft, n)

        # ============================================================
        # 3. For each feature i, remove its contribution and apply N
        # ============================================================
        shapley_values = np.zeros(d, dtype=float)
        onevec = np.ones(n, dtype=float)

        for i in range(d):
            # Remove feature i's contribution:
            # log H_{-i}(ω_ℓ, s) = Σ_{j≠i} log|F_{ℓ j s}| + i Σ_{j≠i} arg(F_{ℓ j s})
            logH_real = total_log_abs - log_abs[:, i, :]  # (n_fft, n)
            logH_imag = total_ang - ang[:, i, :]          # (n_fft, n)

            # Center real part per sample (across ℓ) to avoid overflow/underflow
            base = np.max(logH_real, axis=0)             # (n,)
            logH_centered = logH_real - base[None, :]    # (n_fft, n)

            # Reconstruct H_{-i}(ω_ℓ, s) safely:
            # H_tilde = exp(logH_centered) * exp(i * logH_imag)
            H_tilde = np.exp(logH_centered) * np.exp(1j * logH_imag)  # (n_fft, n)

            # Apply linear functional: result_s = Σ_ℓ N_ℓ H_{-i}(ω_ℓ, s)
            tmp = N @ H_tilde                           # (n,), complex

            # Rescale; clip base to double-precision-safe range
            base_clipped = np.clip(base, -700.0, 700.0)
            scale = np.exp(base_clipped)                # (n,)
            result = (scale * tmp).real.astype(np.float64)  # (n,)

            # Shapley for feature i:
            k_i = kv[i]                                 # (n,)
            shapley_values[i] = self.alpha.dot((k_i - onevec) * result).item()

        return shapley_values

    def explain_by_kernel_vectors_esp(self, kernel_vectors, use_scaling=True):
        """
        Highly stable O(d^2 * m) Shapley computation using ESP prefix/suffix
        WITHOUT forming any 3D arrays.
        This is the most numerically stable method for d up to ~200.

        Parameters
        ----------
        kernel_vectors : list or array, shape (d, n)
            k_j(X_j, x_j) for each feature j.
        use_scaling : bool
            Whether to scale each sample dimension to avoid overflow.

        Returns
        -------
        shap : array, shape (d,)
            Shapley value for each feature.
        """
        import numpy as np

        # ---------------------------------------------------------------
        # Normalize input shape to (d, n)
        # ---------------------------------------------------------------
        if isinstance(kernel_vectors, list):
            K = np.stack(kernel_vectors, axis=0)
        else:
            K = np.asarray(kernel_vectors)
            if K.shape == (self.n, self.d):
                K = K.T
            if K.shape != (self.d, self.n):
                raise ValueError(f"kernel_vectors must have shape (d,n), got {K.shape}")

        d, n = K.shape
        alpha = self.alpha.astype(float)
        mu = self.precompute_mu(d).astype(float)
        ones = np.ones(n, float)

        # ---------------------------------------------------------------
        # Optional scaling for numerical stability: scale each COLUMN.
        # K[j, s]  ->  K[j,s] / scale[s]
        # ESPs computed on scaled inputs, then corrected by scale^q.
        # ---------------------------------------------------------------
        if use_scaling:
            scale = np.maximum(1.0, np.max(np.abs(K), axis=0))
            K_scaled = K / scale
            # Precompute scale^q for q=0..d-1
            # shape = (d, n)
            scale_powers = scale[None, :] ** np.arange(d)[:, None]
        else:
            K_scaled = K
            scale_powers = np.ones((d, n))

        # ---------------------------------------------------------------
        # Compute prefix ESPs L[j,q] for q ≤ j
        # Stored as a 2D array-of-arrays: L[j] is a (j+1, n) array.
        # Total memory O(d*n).
        # ---------------------------------------------------------------
        L = [None] * (d + 1)
        L[0] = np.zeros((1, n), float)
        L[0][0] = 1.0  # e0 = 1

        for j in range(1, d + 1):
            kj = K_scaled[j - 1]
            prev = L[j - 1]
            # new row has size (j+1, n)
            cur = np.zeros((j + 1, n), float)
            cur[0] = 1.0
            # recurrence e_q = prev[q] + kj * prev[q-1]
            for q in range(1, j):
                cur[q] = prev[q] + kj * prev[q - 1]
            cur[j] = kj * prev[j - 1]
            L[j] = cur

        # ---------------------------------------------------------------
        # Compute suffix ESPs R[j,q] for q ≤ d-j
        # R[j] has shape (d-j+1, n)
        # Also O(d*n) memory.
        # ---------------------------------------------------------------
        R = [None] * (d + 1)
        R[d] = np.zeros((1, n), float)
        R[d][0] = 1.0

        for j in range(d - 1, -1, -1):
            kj = K_scaled[j]
            prev = R[j + 1]
            # size (d-j+1, n)
            cur = np.zeros((d - j + 1, n), float)
            cur[0] = 1.0
            k = d - j
            for q in range(1, k):
                cur[q] = prev[q] + kj * prev[q - 1]
            cur[k] = kj * prev[k - 1]
            R[j] = cur

        # ---------------------------------------------------------------
        # For each feature j: convolve L[j] and R[j+1] to get e_minus[j]
        # e_minus[j,q] = sum_{t=0..q} L[j,t] * R[j+1,q-t]
        # ---------------------------------------------------------------
        shap = np.zeros(d, float)

        for j in range(d):
            Lj = L[j]        # shape (j+1, n)
            Rj = R[j + 1]    # shape (d-j, n)
            max_q = d - 1

            # Accumulate ω_j(s) = Σ_q μ_q e_minus[j,q](s)
            result = np.zeros(n, float)

            # Convolution per degree q
            # q runs 0..d-1
            # t runs max(0, q - (d-(j+1))) .. min(q, j)
            for q in range(max_q + 1):
                t_min = max(0, q - (d - (j + 1)))
                t_max = min(q, j)
                eq = np.zeros(n, float)

                for t in range(t_min, t_max + 1):
                    eq += Lj[t] * Rj[q - t]

                if use_scaling:
                    eq *= scale_powers[q]

                result += mu[q] * eq

            shap[j] = alpha.dot((K[j] - ones) * result)

        return shap

    def explain_by_kernel_vectors_esp_quadratic(self, kernel_vectors, use_scaling=True):
        """
        Exact and numerically stable O(d^2 * n) Shapley computation.

        Key idea:
        - Compute prefix ESPs L[j,t] for t=0..j
        - Compute μ-weighted suffix DP arrays G[j,t] for t=0..d-1 via
                G[j,t] = G[j+1,t] + K_scaled[j] * G[j+1,t+1]
            where G[d,t] = μ_t * 1  (and μ_d = 0 sentinel).
        - Then for each feature j,
                ω_j = sum_{t=0..min(j,d-j-1)} L[j,t] * G[j+1,t]
            and
                φ_j = αᵀ ((K_j - 1) ⊙ ω_j)

        No Vandermonde, no FFT, no 3D arrays.
        """

        import numpy as np

        # ---------------------------------------------------------------
        # Normalize input shape to (d, n)
        # ---------------------------------------------------------------
        if isinstance(kernel_vectors, list):
            K = np.stack(kernel_vectors, axis=0)
        else:
            K = np.asarray(kernel_vectors)
            if K.shape == (self.n, self.d):
                K = K.T
            if K.shape != (self.d, self.n):
                raise ValueError(f"kernel_vectors must have shape (d,n), got {K.shape}")

        d, n = K.shape
        alpha = self.alpha.astype(float)
        ones = np.ones(n, float)

        # mu[q] for q=0..d-1
        mu = self.precompute_mu(d).astype(float)
        if mu.shape[0] < d:
            raise ValueError(f"precompute_mu(d) must return length >= d, got {mu.shape}")

        # ---------------------------------------------------------------
        # Optional scaling for numerical stability: scale each column s
        # K[j,s] -> K[j,s] / scale[s]
        # Then: e_q(original) = scale^q * e_q(scaled)
        # ---------------------------------------------------------------
        if use_scaling:
            scale = np.maximum(1.0, np.max(np.abs(K), axis=0))
            K_scaled = K / scale
            # scale_powers[q,s] = scale[s]^q  for q=0..d-1
            scale_powers = scale[None, :] ** np.arange(d)[:, None]
        else:
            K_scaled = K
            scale_powers = None

        # ---------------------------------------------------------------
        # Prefix ESPs L: L[j] has shape (j+1, n)
        # L[j,t] = e_t over features {0,...,j-1} computed on K_scaled
        # ---------------------------------------------------------------
        L = [None] * (d + 1)
        L[0] = np.zeros((1, n), float)
        L[0][0] = 1.0

        for j in range(1, d + 1):
            kj = K_scaled[j - 1]
            prev = L[j - 1]
            cur = np.zeros((j + 1, n), float)
            cur[0] = 1.0
            for t in range(1, j):
                cur[t] = prev[t] + kj * prev[t - 1]
            cur[j] = kj * prev[j - 1]
            L[j] = cur

        # ---------------------------------------------------------------
        # μ-weighted suffix DP G
        #
        # Define (on scaled K):
        #   G[j,t](s) = sum_{u>=0} μ_{t+u} * e_u( K_scaled[j+1:], sample s )
        #
        # Recurrence:
        #   G[j,t] = G[j+1,t] + K_scaled[j] * G[j+1,t+1]
        #
        # We store all G[j] as shape (d+1, n) for shifts t=0..d
        # where row d is the sentinel (all zeros).
        #
        # Base:
        #   G[d,t] = μ_t  (since suffix is empty -> e_0=1, e_{u>0}=0)
        # ---------------------------------------------------------------
        G = [None] * (d + 1)
        G[d] = np.zeros((d + 1, n), float)
        for t in range(d):
            G[d][t] = mu[t] * ones
        # G[d][d] = 0 sentinel

        for j in range(d - 1, -1, -1):
            kj = K_scaled[j]
            G_next = G[j + 1]
            cur = np.zeros((d + 1, n), float)
            # t=0..d-1 safe because G_next[t+1] exists up to t=d-1 (uses sentinel at d)
            for t in range(d):
                cur[t] = G_next[t] + kj * G_next[t + 1]
            # cur[d] remains 0 sentinel
            G[j] = cur

        # ---------------------------------------------------------------
        # Shapley computation
        #
        # ω_j(s) = sum_q μ_q * e_q(K_{-j})(s)
        # Using the factorization:
        #   ω_j = sum_t L[j,t] * H[j,t]
        # where H[j,t] = sum_u μ_{t+u} * e_u(suffix after j)
        #
        # In our DP storage, for feature j:
        #   H[j,t] = (correctly scaled) G[j+1, t] with scaling adjustment.
        #
        # Scaling adjustment:
        #   prefix term L[j,t] corresponds to degree t -> multiply by scale^t
        #   suffix u term is already absorbed in G via μ_{t+u}, but overall degree is (t+u),
        #   which we need to correct by scale^(t+u).
        #
        # To do this without expanding u, we "scale-correct" μ into μ_scaled:
        #   μ_scaled[q,s] = μ_q * scale[s]^q
        #
        # Then running the DP with μ_scaled makes G already corrected,
        # and we only need to scale-correct prefix degree t? (No: prefix already uses K_scaled)
        #
        # Therefore we implement the *simple* correct approach:
        #   Build μ_scaled[q,s] = μ_q * scale^q
        #   Rebuild G using μ_scaled (vector per q), then omega uses L (no extra powers)
        # ---------------------------------------------------------------

        if use_scaling:
            # Build μ_scaled as matrix (d+1, n): μ_scaled[q,s] = μ_q * scale[s]^q
            mu_scaled = np.zeros((d + 1, n), float)
            for q in range(d):
                mu_scaled[q] = mu[q] * scale_powers[q]
            # sentinel row d is 0

            # Recompute G with μ_scaled base (same recurrence)
            G[d] = mu_scaled.copy()
            for j in range(d - 1, -1, -1):
                kj = K_scaled[j]
                G_next = G[j + 1]
                cur = np.zeros((d + 1, n), float)
                for t in range(d):
                    cur[t] = G_next[t] + kj * G_next[t + 1]
                G[j] = cur

        # Now compute shap values
        shap = np.zeros(d, float)

        for j in range(d):
            Lj = L[j]          # (j+1, n)
            Gj = G[j + 1]      # (d+1, n), shifts t

            omega = np.zeros(n, float)
            # Valid t must satisfy:
            #   prefix degree t ≤ j
            #   suffix degree u exists up to d-j-1, but that is already embedded in Gj
            # The only hard cap is t ≤ j; Gj[t] beyond feasible suffix yields 0 anyway.
            for t in range(j + 1):
                omega += Lj[t] * Gj[t]

            shap[j] = alpha.dot((K[j] - ones) * omega)

        return shap

class RBFLocalExplainer(ProductKernelLocalExplainer):
    
    def __init__(self, model, value_function: str = "observational"):
        """
        Initialize the Shapley Value Explainer.

        Args:
            model: A scikit-learn model (GP, SVM or SVR) with RBF kernel
            value_function: Which value function to use. Options:
                - "observational" (default): interventional value function
                  v_x(S) = sum_i alpha_i prod_{j in S} k_j(x_j,x_i_j) prod_{j notin S} nu_j^{(i)}.
                - "masking": baseline/masking value function v_x(S) = alpha^T k_S(X_S, x_S).
        """
        super().__init__(model)
        self.gamma = self.get_gamma()
        self.value_function = value_function

        # Keep a copy of the raw model coefficients; we may reweight for observational mode.
        self.alpha_raw = self.alpha.copy()

        # Precompute nu_j^{(i)} = E_{X_j}[k_j(X_j, x_j^{(i)})] empirically from training set.
        # Shape: (n_samples, n_features). Used when value_function == 'observational'.
        self.nu = self._precompute_nu(self.X_train)

        # For observational value function, reweight alpha as:
        # 	ilde{alpha}_i = alpha_i * prod_j nu_j^{(i)}.
        if self.value_function == "observational":
            prod_nu = np.prod(self.nu, axis=1)
            self.null_game = self.alpha_raw @ prod_nu
            self.alpha = self.alpha_raw * prod_nu
        else:
            self.alpha = self.alpha_raw

    def get_gamma(self):
        """
        Retrieve the gamma parameter based on the model type.

        Returns:
            Gamma parameter for the RBF kernel.
        """
        if hasattr(self.model, "_gamma"):  # For SVM/SVR
            return self.model._gamma
        
        if hasattr(self.model, "gamma"):  # For KRR
            if self.model.gamma is not None:
                return self.model.gamma
            elif self.model.get_params()['kernel'] == 'rbf':
                return 1.0 / self.model.X_fit_.shape[1]

        elif hasattr(self.model.kernel_, "length_scale"):  # For GP (kernel_ has the posterior kernel fitted to the data)
            return (2 * (self.model.kernel_.length_scale ** 2)) ** -1
        

        else:
            raise ValueError("Unsupported model type for Shapley value computation.")

    def compute_kernel_vectors(self, X, x):
        """
        Compute kernel vectors for a given dataset X and instance x.

        Args:
            X: The dataset (2D array) used to train the model.
            x: The instance (1D array) for which to compute Shapley values.

        Returns:
            List of kernel vectors corresponding to each feature. Length = number of features.
        """

        # Initialize the kernel matrix
        kernel_vectors = []

        # For each feature j, compute base kernel vector k_j(X[:,j], x_j).
        for j in range(self.d):
            k_vec = rbf_kernel(
                X[:, j].reshape(-1, 1),
                x[..., np.newaxis][j].reshape(1, -1),
                gamma=self.gamma,
            ).squeeze()

            if self.value_function == "observational":
                # Use transformed kernel vectors: \tilde{k}_j(i) = k_j(i) / nu_j^{(i)}.
                # Together with alpha reweighting in __init__, this implements
                # v_x(S) in observational/interventional form via the masking machinery.
                denom = self.nu[:, j]
                # Safe divide (RBF is strictly positive, but guard just in case)
                k_vec = k_vec / np.maximum(denom, 1e-300)

            kernel_vectors.append(k_vec)

        return kernel_vectors

    def _precompute_nu(self, X):
        """
        Empirically estimate nu_j^{(i)} = E_{X_j}[k_j(X_j, x_j^{(i)})] for all features j and
        all training points i, using the training set marginals.

        For RBF base kernels, this equals the mean over the j-th feature's Gram matrix columns.

        Args:
            X: Training data matrix of shape (n_samples, n_features).

        Returns:
            nu: np.ndarray of shape (n_samples, n_features) with nu[i, j].
        """
        n, d = X.shape
        nu = np.empty((n, d), dtype=float)
        for j in range(d):
            G = rbf_kernel(X[:, j].reshape(-1, 1), X[:, j].reshape(-1, 1), gamma=self.gamma)
            # nu_j^{(i)} is the empirical expectation over X_j: mean over rows (or columns) for fixed i.
            nu[:, j] = G.mean(axis=0)
        return nu

    def _compute_shapley_value(self, kernel_vectors, feature_index):
        """
        Compute the Shapley value for a specific feature of an instance.

        Args:
            X: The dataset (2D array) used to train the model.
            x: The instance (1D array) for which to compute the Shapley value.
            feature_index: Index of the feature.

        Returns:
            Shapley value for the specified feature.
        """
        
        alpha = self.alpha 
        cZ_minus_j = [kernel_vectors[i] for i in range(self.d) if i != feature_index]
        e_polynomials = self.compute_elementary_symmetric_polynomials(cZ_minus_j)
        mu_coefficients = self.precompute_mu(self.d)
        
        # Compute kernel vector for the chosen feature
        k_j = kernel_vectors[feature_index]
        onevec = np.ones_like(k_j)
        
        # Compute the Shapley value
        result = np.zeros_like(k_j)
        for q in range(self.d):
            if q < len(e_polynomials):
                result += mu_coefficients[q] * e_polynomials[q]
        
        shapley_value = alpha.dot((k_j - onevec) * result)

        
        return shapley_value.item()
    
    def explain(self, x):
        """
        Compute Shapley values for all features of an instance.

        Args:
            X: The dataset (2D array) used to train the model.
            x: The instance (1D array) for which to compute Shapley values.

        Returns:
            List of Shapley values, one for each feature.
        """
        
        import time
        kernel_vectors = self.compute_kernel_vectors(self.X_train, x)
        shapley_values = []
        # kernel_vectors = np.array(np.array(kernel_vectors)*3).tolist()
        
        start_time = time.time()
        shap_vals_esp = self.explain_by_kernel_vectors_esp_quadratic(kernel_vectors)
        print(f"Time taken to compute Shapley value with ESP: {time.time() - start_time} seconds")

        start_time = time.time()
        for j in range(self.d):
            shapley_values.append(self._compute_shapley_value(kernel_vectors, j))
        print(f"Time taken to compute Shapley value: {time.time() - start_time} seconds")

        shap_vals = self.explain_by_kernel_vectors_fft(kernel_vectors)
        shap_vals2 = self.explain_by_kernel_vectors_fft2(kernel_vectors)

        shap_vals_log = self.explain_by_kernel_vectors_fft_log(kernel_vectors)
        shap_vals_log2 = self.explain_by_kernel_vectors_fft_log_direct(kernel_vectors)

        shapley_values2 = self.explain_by_kernel_vectors_chebyshev(kernel_vectors)

        start_time = time.time()
        shap_gausleg = weighted_values_gauss_legendre(np.asarray(kernel_vectors)-1, self.alpha, len(x) // 2)
        print(f"Time taken for Gauss-Legendre Shapley value: {time.time() - start_time} seconds")

        # start_time = time.time()
        # shap_gauslejax = weighted_values_gl_jax_auto(np.asarray(kernel_vectors)-1, self.alpha, 50)
        # print(f"Time taken for Gauss-Legendre Shapley value: {time.time() - start_time} seconds")

        E2 = self.compute_elementary_symmetric_polynomials(kernel_vectors)
        start_time = time.time()
        E, shap_parallel = weighted_values_from_kernel_vectors(np.asarray(kernel_vectors), self.precompute_mu(self.d), self.alpha, np.array(E2))
        print(f"Time taken for parallel Shapley value: {time.time() - start_time} seconds")


        # start_time = time.time()
        # banzhaf_vals = unweighted_values_from_kernel_vectors(np.asarray(kernel_vectors), self.alpha)
        # print(f"Time taken for parallel Banzhaf value: {time.time() - start_time} seconds")

        # Additional: compute Shapley via reusable ESPComputer for comparison
        esp = ESPComputer(method="quadratic_stable", use_scaling=True, renorm=True)
        Omega = esp.compute_weight_vectors(kernel_vectors)
        K = np.stack(kernel_vectors, axis=0)
        onevec = np.ones(self.n, dtype=K.dtype)
        shap_via_esp_obj = np.array([self.alpha.dot((K[j] - onevec) * Omega[j]) for j in range(self.d)])
        print(f"Time taken for ESPComputer Shapley value: {time.time() - start_time} seconds")

        return shapley_values
    
    def explain_brute_force(self, x):
        """
        Compute Shapley values for all features of an instance using brute-force method.

        Args:
            kernel_vectors: np.ndarray of shape (n, d), kernel values for each training point and feature.

        Returns:
            List of Shapley values, one for each feature.
        """
        
        kernel_vectors = self.compute_kernel_vectors(self.X_train, x)
        return self.brute_force_shapley(np.array(kernel_vectors).T)


# Example Usage
if __name__ == "__main__":
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC, SVR
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler

    # Generate a synthetic regression dataset with 10 features
    X, y = make_regression(n_samples=1000, n_features=100, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Standardize the features
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # # Train an SVR model with RBF kernel
    # svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')
    # svr_model.fit(X_train, y_train)

    # # train a GP
    kernel =  RBF(1.0, (1e-3, 1e3))
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
    model.fit(X_train, y_train)


    # Initialize the explainer with this model
    explainer = RBFLocalExplainer(model, value_function="product")

    # Test instance
    x = X_test[0]  # Instance to explain

    # Compute Shapley values
    shapley_values = explainer.explain(x)
    print("Shapley Values:", shapley_values)


    # Train a Kernel Ridge Regression model with RBF kernel
    krr_model = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.1)
    krr_model.fit(X_train, y_train)

    # Initialize the explainer with the KRR model
    explainer_krr = RBFLocalExplainer(krr_model)

    # Test instance
    x_krr = X_test[0]

    # Compute Shapley values for KRR
    shapley_values_krr = explainer_krr.explain(x_krr)
    print("Shapley Values (KRR):", shapley_values_krr)
    print(f"sum of Shapley values (KRR): {sum(shapley_values_krr)}")
    print("Predicted value (KRR):", krr_model.predict([x_krr])[0])

    # shap_vals = explainer.explain_brute_force(x)

    # print(f"sum of Shapley values is: {sum(shapley_values)}")

        
        # ------------------------- Classification Example -------------------------
    from sklearn.datasets import make_classification
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Generate a synthetic classification dataset (binary classification)
    X_clf, y_clf = make_classification(n_samples=200, n_features=70, n_informative=5, 
                                    n_redundant=2, n_classes=2, random_state=42)  # <sup data-citation="6" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://machinelearningmastery.com/gaussian-processes-for-classification-with-python/">6</a></sup>

    # Split the data into training and testing sets
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

    # Standardize the features
    scaler_clf = StandardScaler()
    X_train_clf = scaler_clf.fit_transform(X_train_clf)
    X_test_clf = scaler_clf.transform(X_test_clf)

    # Train an SVC model with RBF kernel (set probability=True to enable probability estimates)
    svc_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, random_state=42)
    svc_model.fit(X_train_clf, y_train_clf)  # <sup data-citation="4" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://www.geeksforgeeks.org/support-vector-regression-svr-using-linear-and-non-linear-kernels-in-scikit-learn/">4</a></sup>

    # Initialize the explainer with the chosen classifier (e.g., the GP classifier)
    explainer_clf = RBFLocalExplainer(svc_model)  # use same explainer interface as for regression

    # Test instance for classification
    x_clf = X_test_clf[0]  # instance to explain

    # Compute Shapley values for classification
    shapley_values_clf = explainer_clf.explain(x_clf)
    print("Shapley Values (Classification):", shapley_values_clf)

    # You can also observe the predicted probability and the predicted class:
    print(f"sum of Shapley vlaue {sum(shapley_values_clf)}")
    print("predicted decision function: ", svc_model.decision_function([x_clf])[0])
    print("intercept is: ", svc_model.intercept_)



    # Alternatively, train a Gaussian Process Classifier with an RBF kernel
    kernel = RBF(1.0, (1e-3, 1e3))
    gpc = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gpc.fit(X_train_clf, y_train_clf)  # 

    # Initialize the explainer with the chosen classifier (e.g., the GP classifier)
    explainer_clf = RBFLocalExplainer(gpc)  # use same explainer interface as for regression

    # Test instance for classification
    x_clf = X_test_clf[0]  # instance to explain

    # Compute Shapley values for classification
    shapley_values_clf = explainer_clf.explain(x_clf)
    print("Shapley Values (Classification):", shapley_values_clf)

    # You can also observe the predicted probability and the predicted class:
    print("Predicted probabilities:", gpc.predict_proba([x_clf])[0])
    print("Predicted class:", gpc.predict([x_clf])[0])
