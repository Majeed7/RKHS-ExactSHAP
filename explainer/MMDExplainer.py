import numpy as np
import math
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist, cdist
from joblib import Parallel, delayed, parallel_config

class MMDExplainer:
    def __init__(self, X, Z, **kwargs):
        """
        Initialize the MMD Explainer using keyword arguments.
        
        Args:
            X: Input feature matrix of shape (n_samples, n_features).
            Z: Comparison feature matrix of shape (m_samples, n_features).
            kernel_x: Kernel type for features ('rbf' supported). Default 'rbf'.
            gamma_x: Bandwidth parameter for feature RBF kernels. If None, use median heuristic.
        """
        self.X = X
        self.Z = Z
        self.n, self.d = self.X.shape
        self.m = self.Z.shape[0]
        
        self.estimation_type = kwargs.get("estimation_type", "U")
        self.coef_xz = self.n * self.m
        if self.estimation_type == "U":
            self.coef_x = self.n * (self.n - 1)
            self.coef_z = self.m * (self.m - 1)
            
        elif self.estimation_type == "V":
            self.coef_x = self.n ** 2
            self.coef_z = self.m ** 2

        self.gamma = kwargs.get("gamma", None)
        
        if self.gamma is None:
             self.gamma = self.compute_median_heuristic(np.vstack([self.X, self.Z]))

        # self.gamma_x = kwargs.get("gamma_x", None)
        # self.gamma_z = kwargs.get("gamma_z", None)
        # self.gamma_xz = kwargs.get("gamma_xz", None)
        
        # if self.gamma_x is None:
        #     self.gamma_x = self.compute_median_heuristic(self.X)
        
        # if self.gamma_z is None:
        #     self.gamma_z = self.compute_median_heuristic(self.Z)

        # if self.gamma_xz is None:
        #     self.gamma_xz = self.compute_median_heuristic(self.X, self.Z)
        
        self.KXs = self._compute_kernels(self.X, self.gamma)
        self.KZs = self._compute_kernels(self.Z, self.gamma)
        self.KXZs = self._compute_cross_kernel(self.X, self.Z, self.gamma)
        
        self.mu = self.precompute_mu(self.d)

    def compute_median_heuristic(self, X, Z=None):
        if Z is None:
            pairwise_distances = pdist(X, metric='euclidean')
        else:
            pairwise_distances = cdist(X, Z, metric='euclidean')

        sigma = np.median(pairwise_distances)
        gamma = 1.0 / (2.0 * (sigma ** 2)) if sigma != 0 else 1.0
        return gamma

    def _compute_kernels(self, X, gamma):
        #with parallel_config(backend='loky', inner_max_num_threads=1):
        Ks = Parallel(n_jobs=-1)(
            delayed(self._compute_feature_kernel)(X[:, j].reshape(-1, 1), gamma) for j in range(self.d)
        )
        return Ks

    def _compute_feature_kernel(self, Xj, gamma):
        K = rbf_kernel(Xj, gamma=gamma)
        return K

    def _compute_cross_kernel(self, X, Z, gamma):
        KXZs = Parallel(n_jobs=-1)(
            delayed(self._compute_feature_cross_kernel)(X[:, j].reshape(-1, 1), Z[:, j].reshape(-1, 1), gamma) for j in range(self.d)
        )
        return KXZs

    def _compute_feature_cross_kernel(self, Xj, Zj, gamma):
        return rbf_kernel(Xj, Zj, gamma=gamma)
        
    def precompute_mu(self, d):
        unnormalized = [math.factorial(q) * math.factorial(d - q - 1) for q in range(d)]
        return np.array(unnormalized) / math.factorial(d)
    
    def compute_elementary_symmetric_polynomials(self, KXs, KZs, KXZs):
        e_X = self._compute_esp(KXs)
        e_Z = self._compute_esp(KZs)
        e_XZ = self._compute_esp(KXZs)
        return e_X, e_Z, e_XZ

    def _compute_esp_recursive(self, kernel_matrices):
        s = [sum([np.power(k, p) for k in kernel_matrices]) for p in range(0, len(kernel_matrices) + 1)]
        e = [np.ones_like(kernel_matrices[0])]
        for r in range(1, len(kernel_matrices) + 1):
            term = 0
            for k in range(1, r + 1):
                term += ((-1) ** (k - 1)) * e[r - k] * s[k]
            e.append(term / r)
        return e
    
    def _compute_esp(self, kernel_matrices):
        """
        Compute elementary symmetric polynomials (ESPs) using a numerically stable
        dynamic programming approach.
        """
        # Ensure float64 precision for stability
        kernel_matrices = [np.array(k, dtype=np.float64) for k in kernel_matrices]
        
        # Handle empty input edge case
        if not kernel_matrices:
            return []
        
        # Normalization to prevent overflow
        max_abs_k = max(np.max(np.abs(k)) for k in kernel_matrices) or 1.0
        scaled_kernel = [k / max_abs_k for k in kernel_matrices]
        
        # Initialize polynomial coefficients: e[degree] = coefficient
        # Start with e_0 = 1 (constant term)
        e = [np.ones_like(scaled_kernel[0], dtype=np.float64)]
        
        # Sequentially build the polynomial (x - k1)(x - k2)...(x - kn)
        for k in scaled_kernel:
            new_e = [np.zeros_like(e[0]) for _ in range(len(e) + 1)]
            
            # Constant term: -k * previous_constant_term
            new_e[0] = -k * e[0]
            
            # Middle terms: new_e[i] = previous_e[i-1] - k * previous_e[i]
            for i in range(1, len(e)):
                new_e[i] = e[i-1] - k * e[i]
            
            # Highest degree term: x^m term (always 1 in our construction)
            new_e[-1] = e[-1].copy()
            
            e = new_e
        
        # Extract ESPs from polynomial coefficients
        n = len(scaled_kernel)
        elementary = [np.ones_like(e[0])]  # e_0 = 1
        
        for r in range(1, n + 1):
            # Extract coefficient for x^{n-r} and apply sign correction
            sign = (-1) ** r
            scaled_e_r = sign * e[n - r]
            
            # Reverse the normalization scaling
            elementary_r = scaled_e_r * (max_abs_k ** r)
            elementary.append(elementary_r)
        
        return elementary

    def explain(self):
        shapley_values = np.zeros(self.d)
        
        onevec_X = np.ones_like(self.KXs[0])
        onevec_Z = np.ones_like(self.KZs[0])
        onevec_XZ = np.ones_like(self.KXZs[0])

        if self.estimation_type == "U":
            np.fill_diagonal(onevec_Z, 0)   
            np.fill_diagonal(onevec_X, 0)
            [np.fill_diagonal(t, 0) for t in self.KXs]
            [np.fill_diagonal(t, 0) for t in self.KZs]
 
        
        def shapley_value_j(j):
            K_j_X = self.KXs[j]
            K_j_Z = self.KZs[j]
            K_j_XZ = self.KXZs[j]
            
            KX_minus_j = [self.KXs[i] for i in range(self.d) if i != j]
            KZ_minus_j = [self.KZs[i] for i in range(self.d) if i != j]
            KXZ_minus_j = [self.KXZs[i] for i in range(self.d) if i != j]
            
            e_polynomials_X, e_polynomials_Z, e_polynomials_XZ = self.compute_elementary_symmetric_polynomials(KX_minus_j, KZ_minus_j, KXZ_minus_j)

            psi_X = sum(self.mu[q] * e_polynomials_X[q] for q in range(self.d))
            psi_Z = sum(self.mu[q] * e_polynomials_Z[q] for q in range(self.d))
            psi_XZ = sum(self.mu[q] * e_polynomials_XZ[q] for q in range(self.d))
            
            result_X = (K_j_X - onevec_X) * psi_X
            result_Z = (K_j_Z - onevec_Z) * psi_Z
            result_XZ = (K_j_XZ - onevec_XZ) * psi_XZ
            
            if self.estimation_type == "U":
                assert np.sum(np.diag(result_X)) == 0, "Diagonal elements should be zero."
                assert np.sum(np.diag(result_Z)) == 0, "Diagonal elements should be zero." 

            X_contribution = np.sum(result_X) / self.coef_x
            Z_contribution = np.sum(result_Z) / self.coef_z 
            XZ_contribution = np.mean(result_XZ) # np.sum(result_XZ) / self.coef_xz

            shapley_value_j = X_contribution + Z_contribution - (2 * XZ_contribution)
            
            return shapley_value_j.item()

        shapley_values = Parallel(n_jobs=-1)(
            delayed(shapley_value_j)(j) for j in range(self.d)
        )
        
        return shapley_values

# Example Usage
if __name__ == "__main__":
    
    np.random.seed(42)
    X = np.random.multivariate_normal(mean=[2, 3, 7, 1, 0], cov=np.eye(5), size=1000)
    Z = np.random.multivariate_normal(mean=[5, 3, 2, 5, 0], cov=np.eye(5), size=1000)

    explainer = MMDExplainer(X=X, Z=Z)
    sv = explainer.explain()
    print("MMD Shapley Values:", sv)
    print("Sum of Shapley values:", np.sum(sv))

    K_X = rbf_kernel(X, gamma = explainer.gamma)
    K_Z = rbf_kernel(Z, gamma = explainer.gamma)
    K_XZ = rbf_kernel(X, Z, gamma = explainer.gamma)
    X_contribution = ((explainer.n * (explainer.n-1) )**-1) * ( np.sum(K_X) - np.sum(np.diag(K_X)) )
    Z_contribution = ((explainer.m * (explainer.m-1) )**-1) * ( np.sum(K_Z) - np.sum(np.diag(K_Z)) )
    XZ_contribution = 2 * np.mean(K_XZ)
    mmd_estimate =  X_contribution + Z_contribution - XZ_contribution 
