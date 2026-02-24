import numpy as np
import math
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed
from sklearn.utils.multiclass import type_of_target

try:
    from .esp import ESPComputer
except ImportError:
    from esp import ESPComputer

class HSICExplainer:
    def __init__(self, X, y, **kwargs):
        """
        Initialize the HSIC Explainer using keyword arguments.
        
        Expected keys in kwargs:
            X: Input feature matrix of shape (n_samples, n_features).
            y: Target variable of shape (n_samples,).
            kernel_x: Kernel type for features ('rbf' supported). Default 'rbf'.
            kernel_y: Kernel type for target ('rbf' supported). Default 'rbf' for regression and delta for classification.
            gamma_x: Bandwidth parameter for feature RBF kernels. If None,
                     use median heuristic. 
            gamma_y: Bandwidth parameter for target RBF kernel. If None, use median heuristic.
        """
        self.X = X
        self.y = y
        # Ensure y is 2D
        self.y = self.y.reshape(-1, 1)
        self.n, self.d = self.X.shape

        # Value function selection
        # - "observational" (default): interventional value function from the paper
        #   v(S) = C_{-S} * (1/n^2) tr(K_S H L H), where c_j = E[k_j(X_j,X'_j)] and C_{-S}=prod_{j notin S} c_j.
        # - "baseline1": legacy baseline=1 game used in the previous implementation.
        self.value_function = kwargs.get("value_function", "observational")

        # ESP implementation selection (uses explainer/esp.py)
        self.esp_method = kwargs.get("esp_method", "quadratic")
        self.esp_kwargs = kwargs.get("esp_kwargs", {}) or {}
        self._esp = ESPComputer(method=self.esp_method, **self.esp_kwargs)

        self.kernel_y = kwargs.get("kernel_y", "rbf")
        self.gamma_x = kwargs.get("gamma_x", None)
        self.gamma_y = kwargs.get("gamma_y", None)

        if self.gamma_x is None:
            # Compute a single gamma for all features using the median heuristic
            # applied to the full feature matrix `X` (common RBF heuristic).
            self.gamma_x = self.compute_median_heuristic(self.X)

        if self.kernel_y == "rbf" and self.gamma_y is None:
            self.gamma_y = self.compute_median_heuristic(self.y)

        # Centering matrix H = I - (1/n) * ones
        self.H = np.eye(self.n) - np.ones((self.n, self.n)) / self.n

        # Compute feature kernels using joblib parallel processing
        self.Ks = self._compute_feature_kernels()

        # Compute target kernel
        self.L = self._compute_target_kernel()

        # Precompute HLH = H @ L @ H
        self.HLH = self.H @ self.L @ self.H
        
        self.K = np.prod(np.array(self.Ks), axis=0)
        self.total_HSIC = np.trace(self.HLH @ self.K) / (self.n) ** 2

        # For the observational game, precompute c_j, C and the corresponding null-game value.
        if self.value_function not in {"observational", "baseline1"}:
            raise ValueError(f"Unknown value_function: {self.value_function}")

        if self.value_function == "observational":
            self._precompute_c_constants()
        else:
            self.c = None
            self.C_total = None
            self.null_game = 0.0

        # Keep mu around for backwards compatibility, even though ESPComputer computes its own.
        self.mu = self.precompute_mu(self.d)

    def _precompute_c_constants(self) -> None:
        """Precompute c_j and C for the observational/interventional HSIC value function.

        We estimate c_j = E[k_j(X_j, X'_j)] using the empirical distribution (with replacement),
        which corresponds to averaging all entries of the feature kernel matrix K_j.
        """
        self.c = np.array([float(np.mean(Kj)) for Kj in self.Ks], dtype=np.float64)
        eps = 1e-300
        self.C_total = float(np.prod(np.maximum(self.c, eps)))

        # Baseline (null coalition) value v(∅) = C * (1/n^2) tr(J HLH).
        # With centering, 1^T HLH 1 = 0, so this should be (numerically) ~0.
        ones = np.ones((self.n, 1), dtype=np.float64)
        self.null_game = float(self.C_total * (ones.T @ self.HLH @ ones) / (self.n**2))

    def _compute_feature_kernels(self):
        """Compute RBF kernel matrix for each feature using median heuristic.
           Uses joblib for parallel computation.
        """
        Ks = Parallel(n_jobs=-1)(
            delayed(self._compute_feature_kernel)(j) for j in range(self.d)
        )
        return Ks

    def _compute_feature_kernel(self, j):
        """
            Compute the RBF kernel matrix for feature j
        """
        Xj = self.X[:, j].reshape(-1, 1)
        # Support either a scalar gamma (shared) or a per-feature array of gammas.
        if isinstance(self.gamma_x, (list, np.ndarray)):
            gamma = float(self.gamma_x[j])
        else:
            gamma = float(self.gamma_x)

        return rbf_kernel(Xj, gamma=gamma)

    def _compute_target_kernel(self):
        """
        Compute the kernel matrix for target variable y.
        Uses:
        - A delta kernel (1 if labels match, 0 otherwise) for classification problems.
        - An RBF kernel for regression problems.
        
        The decision is made using sklearn's type_of_target:
        - If y is binary, multiclass, or multilabel-indicator, a classification problem is assumed.
        - Otherwise, it is considered a regression problem.
        
        For the RBF kernel:
        If gamma_y is None, it is computed using the median heuristic over all pairwise Euclidean distances.
        
        Args:
            y : array-like of shape (n_samples,)
                The target variable.
            gamma_y : float, optional
                The RBF kernel gamma parameter for y. If None (and task is regression),
                it is computed using the median heuristic.
        
        Returns:
            K_y : ndarray of shape (n_samples, n_samples)
                The kernel matrix for y.
        """
        # Ensure y is a column vector.
        target_type = type_of_target(self.y)
        
        if target_type in ['binary', 'multiclass', 'multilabel-indicator']:
            # Use delta (Dirac) kernel for classification: 1 if equal, 0 otherwise.
            K_y = (self.y == self.y.T).astype(float)
        else:
            # Regression: use RBF kernel.
            K_y = rbf_kernel(self.y, gamma=self.gamma_y)
        return K_y

    def compute_median_heuristic(self, X):
        """
        Compute the median heuristic and return the RBF `gamma` parameter.

        The function computes the median of pairwise squared Euclidean distances
        (the common variant of the median heuristic) and returns
        gamma = 1 / (2 * median_sq_dist).

        Args:
            X: 2D array of shape (n_samples, n_dim) -- often (n,1) for a single feature

        Returns:
            gamma: float, the RBF kernel `gamma` value to pass to `rbf_kernel`.
        """
        # Use pairwise squared Euclidean distances for the median heuristic.
        pairwise_sq_dists = pdist(X, metric='sqeuclidean')
        median_sq = np.median(pairwise_sq_dists)

        if median_sq <= 0 or np.isnan(median_sq):
            return 1.0

        gamma = 1.0 / (2.0 * float(median_sq))
        return gamma

    def precompute_mu(self, d):
        """
            Precompute mu coefficients for Shapley value computation for d features.
            Formula: mu[q] = (q! * (d - q - 1)!)/ d! for q = 0, ..., d - 1.
        """
        unnormalized = [
            math.factorial(q) * math.factorial(d - q - 1) for q in range(d)
        ]
        return np.array(unnormalized) / math.factorial(d)

    def compute_elementary_symmetric_polynomials_recursive(self, kernel_matrices):
        """Compute elementary symmetric polynomials given a list of kernel matrices.
        
        Returns:
            List of elementary symmetric polynomials e for orders 0 to m.
            e[0] is an array of ones with the same shape as the kernel matrices.
        """

        # Compute power sums
        s = [
            sum([np.power(k, p) for k in kernel_matrices])
            for p in range(0, len(kernel_matrices) + 1)
        ]
        
        # Compute elementary symmetric polynomials
        e = [np.ones_like(kernel_matrices[0])]  # e_0 = 1
        
        for r in range(1, len(kernel_matrices) + 1):
            term = 0 
            for k in range(1, r + 1):
                term += ((-1) ** (k-1)) * e[r - k] * s[k]
            e.append(term / r )
        
        return e
    
    def compute_elementary_symmetric_polynomials(self, kernel_matrices):
        """Compute elementary symmetric polynomials using stable dynamic programming.
        
        Returns:
            List of elementary symmetric polynomials e for orders 0 to m.
            e[0] is an array of ones with the same shape as the kernel matrices.
        """
        
        # Convert to float64 for numerical stability
        kernel_matrices = [np.array(k, dtype=np.float64) for k in kernel_matrices]
        
        # Handle empty input case
        if not kernel_matrices:
            return [np.ones((1,))]  # Default shape if no inputs
        
        # Normalization to prevent overflow
        max_abs = max(np.max(np.abs(k)) for k in kernel_matrices)
        scale_factor = max_abs if max_abs > 0 else 1.0
        scaled_kernels = [k / scale_factor for k in kernel_matrices]
        
        # Initialize polynomial coefficients: e[degree] = coefficient
        # Start with e_0 = 1 (the empty product)
        e = [np.ones_like(scaled_kernels[0])]
        
        # Build polynomial incrementally: (x - k1)(x - k2)...(x - kn)
        for k in scaled_kernels:
            new_e = [np.zeros_like(e[0]) for _ in range(len(e) + 1)]
            
            # Constant term: -k * previous constant term
            new_e[0] = -k * e[0]
            
            # Middle terms: e[i] = previous_e[i-1] - k * previous_e[i]
            for i in range(1, len(e)):
                new_e[i] = e[i-1] - k * e[i]
            
            # Highest degree term (x^m term)
            new_e[-1] = e[-1].copy()
            
            e = new_e
        
        # Extract elementary polynomials with proper scaling and signs
        n = len(scaled_kernels)
        elementary = [np.ones_like(e[0])]  # e_0 = 1
        
        for r in range(1, n + 1):
            # The polynomial coefficients contain (-1)^r e_r at position n-r
            sign = (-1) ** r
            scaled_value = sign * e[n - r]
            
            # Reverse the normalization scaling
            elementary_r = scaled_value * (scale_factor ** r)
            elementary.append(elementary_r)
        
        return elementary

    def explain(self):
        """Compute Shapley values for each feature's contribution to HSIC.
        
        Returns:
            A numpy array with Shapley values for each feature.
        """
        
        # For symmetric kernels (RBF / delta), HLH is symmetric and each feature kernel matrix is symmetric.
        # Since tr(HLH @ A) = sum_{a,b} HLH[a,b] * A[a,b] when A is symmetric, we can compute
        # using only the upper-triangular entries.

        tri_i, tri_j = np.triu_indices(self.n, k=0)
        weights = np.where(tri_i == tri_j, 1.0, 2.0).astype(np.float64)
        B_samples = self.HLH[tri_i, tri_j].astype(np.float64)

        # Build per-entry feature values matrix of shape (num_entries, d)
        K_samples = np.stack([Kj[tri_i, tri_j] for Kj in self.Ks], axis=1).astype(np.float64)

        if self.value_function == "baseline1":
            Omega = self._esp.compute_weight_vectors(K_samples)
            shapley = np.zeros(self.d, dtype=np.float64)
            for j in range(self.d):
                A_samples = (K_samples[:, j] - 1.0) * Omega[:, j]
                shapley[j] = float(np.sum(weights * B_samples * A_samples))
            return shapley

        # Observational/interventional value function from the paper:
        # tilde{K}_j = K_j / c_j, C = prod_j c_j
        # phi_j = (C/n^2) * tr( HLH * ( (tildeK_j - 1) ⊙ sum_q mu(q) e_q(tildeK_{-j}) ) )
        K_tilde_samples = K_samples / self.c[None, :]
        Omega = self._esp.compute_weight_vectors(K_tilde_samples)

        prefactor = float(self.C_total) / float(self.n**2)
        shapley = np.zeros(self.d, dtype=np.float64)
        for j in range(self.d):
            # Compute elementwise contributions and reduce to the trace:
            #
            # - We selected upper-triangular indices (including diagonal) via
            #   `tri_i, tri_j = np.triu_indices(self.n, k=0)`. For each such
            #   (i,j) we built arrays `B_samples = HLH[tri_i, tri_j]` and
            #   `K_tilde_samples[:, j]` (so `A_samples` are the matching
            #   upper-triangle entries of the matrix A := (tildeK_j - 1)
            #   ⊙ (sum_q mu(q) e_q(tildeK_{-j}))).
            #
            # - `weights` equals 1 on the diagonal and 2 off-diagonal. Thus
            #   sum(weights * B_samples * A_samples) =
            #     sum_{i} HLH_{ii} A_{ii} + 2 * sum_{i<j} HLH_{ij} A_{ij}
            #   = sum_{i,j} HLH_{ij} A_{ij} because HLH and A are symmetric.
            #
            # - Finally, sum_{i,j} HLH_{ij} A_{ij} = trace(HLH @ A), so the
            #   weighted sum computed below is an efficient way to compute
            #   trace(HLH @ A) without forming the full matrix A.
            #
            # Therefore the line below implements
            #   prefactor * trace(HLH @ A)
            # with A represented by its upper-triangular samples.
            A_samples = (K_tilde_samples[:, j] - 1.0) * Omega[:, j]
            shapley[j] = prefactor * float(np.sum(weights * B_samples * A_samples))
        return shapley

# Example Usage
if __name__ == "__main__":
    from sklearn.datasets import make_regression, make_classification
    from sklearn.preprocessing import StandardScaler


    # Generate synthetic regression data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten()

    # Create HSICExplainer using keyword arguments
    explainer = HSICExplainer(X=X, y=y)
    sv = explainer.explain()
    print("HSIC Shapley Values:", sv)
    print("Sum of Shapley values:", np.sum(sv))
