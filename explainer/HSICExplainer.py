import numpy as np
import math
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
from sklearn.utils.multiclass import type_of_target

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

        self.kernel_y = kwargs.get("kernel_y", "rbf")
        self.gamma_x = kwargs.get("gamma_x", None)
        self.gamma_y = kwargs.get("gamma_y", None)

        if self.gamma_x is None:
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
        self.total_HSIC = np.trace(self.HLH @ self.K) / (self.n - 1) ** 2
        # Precompute mu coefficients for Shapley value computation (for d features)
        self.mu = self.precompute_mu(self.d)

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
        gamma = self.gamma_x
        
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
        Compute the median heuristic for the RBF kernel bandwidth σ.
        
        Args:
            2D array; either feature Matrix X or target variable y.
            
        Returns:
            sigma: Bandwidth parameter for the RBF kernel.
        """
        # Compute all pairwise Euclidean distances between data points
        pairwise_distances = pdist(X, metric='euclidean')
        
        # Return the median of these distances as the bandwidth σ
        sigma = np.median(pairwise_distances)
        gamma_y = 1 / (2 * (sigma ** 2)) if sigma != 0 else 1.0

        return gamma_y

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
        
        def shapley_value_j(j):
            """
            Compute the Shapley value for feature j.
            """

            # Exclude feature j from the list of feature kernels.
            K_minus_j = [self.Ks[i] for i in range(self.d) if i != j]
            e_polynomials = self.compute_elementary_symmetric_polynomials(K_minus_j)

            # Sum the weighted elementary symmetric polynomials:
            # sum_term = sum_{q=0}^{d-1} mu[q] * e_polynomials[q]
            sum_term = np.zeros_like(self.Ks[j])
            for q in range(self.d):
                if q < len(e_polynomials):
                    sum_term += self.mu[q] * e_polynomials[q]

            # Compute contribution of feature j: (K_j - ones) * sum_term
            K_j = self.Ks[j]
            term = (K_j - np.ones_like(K_j)) * sum_term
            trace_value = np.trace(self.HLH @ term)  # Elementwise product then trace
            
            return trace_value #/ ((self.n - 1) ** 2)


        shapley_values = Parallel(n_jobs=-1)(
            delayed(shapley_value_j)(j) for j in range(self.d)
        )

        # shapley_values = np.zeros(self.d)
        # for j in range(self.d):
        #     shapley_values[j] = shapley_value_j(j)

        return np.array(shapley_values)

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
