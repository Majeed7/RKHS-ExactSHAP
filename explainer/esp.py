import numpy as np

class ESPComputer:
    """
    Computes per-feature weight vectors omega_j(s) = sum_q mu_q * e_q(K_{-j})(s)
    for product-kernel Shapley computations, without multiplying by alpha.

    This object can be reused across explainers: pass in per-feature kernel
    vectors K[j, s] and get back Omega[j, s] needed for Shapley aggregation.

    Methods:
    - compute_weight_vectors(K, method='quadratic', **kwargs) -> (d, n)
    """

    def __init__(self, method: str = "quadratic", use_scaling: bool = True):
        self.method = method
        self.use_scaling = use_scaling

    @staticmethod
    def precompute_mu(d: int) -> np.ndarray:
        """
        Shapley coefficients mu[q] = q!(d-q-1)! / d! for q=0..d-1.
        Stable computation without large factorials:
        mu[q] = 1 / (d * C(d-1, q)). We use the recurrence
            mu[0] = 1/d,
            mu[q+1] = mu[q] * (q+1) / (d-1-q).
        """
        mu = np.empty(d, dtype=np.float64)
        mu[0] = 1.0 / float(d)
        for q in range(d - 1):
            mu[q + 1] = mu[q] * float(q + 1) / float(d - 1 - q)
        return mu

    def compute_weight_vectors(self, kernel_vectors):
        """
        Compute omega vectors for each feature, given per-feature kernel vectors.

        Args:
            kernel_vectors: list or array; shape (d, n) or (n, d).

        Returns:
            Omega: np.ndarray of shape (d, n) where Omega[j,:] are the weights
                   to be elementwise-multiplied with (k_j - 1) and then dotted
                   with alpha by the explainer.
        """
        if self.method == "quadratic":
            return self._esp_weight_vectors_quadratic(kernel_vectors, self.use_scaling)
        elif self.method == "chebyshev":
            return self._esp_weight_vectors_chebyshev(kernel_vectors)
        elif self.method == "fft":
            return self._esp_weight_vectors_fft(kernel_vectors)
        elif self.method == "fft_log":
            return self._esp_weight_vectors_fft_log(kernel_vectors)
        else:
            raise ValueError(f"Unknown ESP method: {self.method}")

    def _normalize_K(self, kernel_vectors):
        if isinstance(kernel_vectors, list):
            K = np.stack(kernel_vectors, axis=0)
        else:
            K = np.asarray(kernel_vectors)
        if K.ndim != 2:
            raise ValueError("kernel_vectors must be 2D (d, n) or (n, d).")
        # Prefer (d, n)
        if K.shape[0] < K.shape[1]:
            # could already be (d, n). We'll return as-is and let caller validate.
            return K
        else:
            # Might be (n, d) if n >> d; check which dimension equals features by context outside.
            # We choose to return orientation that has fewer rows as d if plausible.
            # Safer path: assume if K.shape[0] != d outside, the caller will transpose before calling.
            return K

    def _esp_weight_vectors_quadratic(self, kernel_vectors, use_scaling=True):
                # ---------------------------------------------------------------
        # Normalize input shape to (d, n)
        # ---------------------------------------------------------------
        if isinstance(kernel_vectors, list):
            K = np.stack(kernel_vectors, axis=0)
        else:
            K = np.asarray(kernel_vectors)
            # if K.shape == (self.n, self.d):
            #     K = K.T
            # if K.shape != (self.d, self.n):
            #     raise ValueError(f"kernel_vectors must have shape (d,n), got {K.shape}")

        n, d = K.shape
        K = K.T.astype(float)
        # alpha = self.alpha.astype(float)
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
        Omega = np.zeros((n, d), float)
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

            Omega[:,j] = omega

            # shap[j] = alpha.dot((K[j] - ones) * omega)

        return Omega


    def _esp_weight_vectors_chebyshev(self, kernel_vectors):
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
        if isinstance(kernel_vectors, list):
            K = np.stack(kernel_vectors, axis=0)
        else:
            K = np.asarray(kernel_vectors)
            # if K.shape == (self.n, self.d):
            #     K = K.T
            # if K.shape != (self.d, self.n):
            #     raise ValueError(f"kernel_vectors must have shape (d,n), got {K.shape}")

        n, d = K.shape
        K = K.T.astype(float)

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
        kv_grid = K[None, :, :]          # (1, d, n)
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
        Omega = np.zeros((n, d), dtype=float)
        for i in range(d):
            # H_{-i}(y_ℓ) = Π_{j≠i} (1 + k_j y_ℓ) = P[:, i, :] * S[:, i+1, :]
            H_minus_i = P[:, i, :] * S[:, i + 1, :]    # shape (D+1, n)

            # result(q-aggregated ESP combination) at each training sample:
            # result = Σ_q μ_q e_q  =  N^T H_minus_i (in Chebyshev eval space)
            # N: (D+1,), H_minus_i: (D+1, n) → result: (n,)
            result = N @ H_minus_i   # matmul over ℓ
            Omega[:, i] = result


        return Omega
       

    def _esp_weight_vectors_fft(self, kernel_vectors):
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
        if isinstance(kernel_vectors, list):
            K = np.stack(kernel_vectors, axis=0)
        else:
            K = np.asarray(kernel_vectors)
            # if K.shape == (self.n, self.d):
            #     K = K.T
            # if K.shape != (self.d, self.n):
            #     raise ValueError(f"kernel_vectors must have shape (d,n), got {K.shape}")

        n, d = K.shape
        K = K.T.astype(float)

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
        kv_grid = K[None, :, :]            # (1, d, n)
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
        Omega = np.zeros((n, d), dtype=float)
        for i in range(d):
            # H_{-i}(ω_ℓ, s) = ∏_{j≠i} (1 + k_j(s) ω_ℓ)
            # via prefix/suffix:
            H_minus_i = P[:, i, :] * S[:, i + 1, :]  # (n_fft, n), complex

            # For each sample s, result_s = Σ_q μ_q e_q^{(-i)}(s)
            # = Σ_ℓ N_ℓ H_{-i}(ω_ℓ, s)
            result_complex = N @ H_minus_i          # (n,)
            result = result_complex.real.astype(np.float64)
            Omega[:, i] = result

        return Omega

        
    def _esp_weight_vectors_fft_log(self, kernel_vectors):
        """
        Numerically stable FFT-based Omega computation:
        - Uses roots of unity + FFT for degree weights
        - Performs prefix/suffix in log-space with unwrapped phase
        - Recenters per-sample real logs to avoid overflow
        Returns Omega of shape (d, n).
        """
        import numpy as np

        # Normalize to (d, n)
        if isinstance(kernel_vectors, list):
            kv = np.stack(kernel_vectors, axis=0)
        else:
            kv = np.asarray(kernel_vectors)
        d, n = kv.shape if kv.shape[0] <= kv.shape[1] else (kv.shape[1], kv.shape[0])
        if kv.shape != (d, n):
            kv = kv.T

        # Trivial 1D case
        if d == 1:
            return np.ones((1, n), dtype=float)

        # Roots of unity & FFT weights
        D = d - 1
        n_fft = D + 1
        ell = np.arange(n_fft, dtype=float)
        omega = np.exp(2j * np.pi * ell / n_fft)  # (n_fft,)

        mu = self.precompute_mu(d)
        c = np.zeros(n_fft, dtype=np.complex128)
        c[:D + 1] = mu[:D + 1].astype(np.complex128)
        N = np.fft.fft(c) / n_fft

        # F = 1 + k_j * ω_ℓ
        omega_grid = omega[:, None, None]
        kv_grid = kv[None, :, :]
        F = 1.0 + kv_grid * omega_grid   # (n_fft, d, n), complex

        # Log-space magnitude with epsilon; unwrap phase over features
        eps = 1e-16
        log_abs = np.log(np.abs(F) + eps)             # (n_fft, d, n)
        ang = np.unwrap(np.angle(F), axis=1)          # (n_fft, d, n)

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

        # Assemble Omega with stable rescaling
        Omega = np.zeros((d, n), dtype=np.float64)
        for i in range(d):
            logH_real = logP[:, i, :] + logS[:, i + 1, :]   # (n_fft, n)
            logH_imag = angP[:, i, :] + angS[:, i + 1, :]   # (n_fft, n)

            base = np.max(logH_real, axis=0)                # (n,)
            logH_centered = logH_real - base[None, :]       # (n_fft, n)

            H_tilde = np.exp(logH_centered) * np.exp(1j * logH_imag)
            tmp = N @ H_tilde                               # (n,), complex

            base_clipped = np.clip(base, -700.0, 700.0)
            scale = np.exp(base_clipped)
            result = (scale * tmp).real.astype(np.float64)
            Omega[i] = result

        return Omega
