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
        
        self.gamma_x = kwargs.get("gamma_x", None)
        self.gamma_z = kwargs.get("gamma_z", None)
        self.gamma_xz = kwargs.get("gamma_xz", None)
        
        if self.gamma_x is None:
            self.gamma_x = self.compute_median_heuristic(self.X)
        
        if self.gamma_z is None:
            self.gamma_z = self.compute_median_heuristic(self.Z)

        if self.gamma_xz is None:
            self.gamma_xz = self.compute_median_heuristic(self.X, self.Z)
        
        self.KXs = self._compute_kernels(self.X, self.gamma_x)
        self.KZs = self._compute_kernels(self.Z, self.gamma_z)
        self.KXZs = self._compute_cross_kernel(self.X, self.Z, self.gamma_xz)
        
        self.mu = self.precompute_mu(self.d)

    def compute_median_heuristic(self, X, Z=None):
        if Z is None:
            pairwise_distances = pdist(X, metric='euclidean')
        else:
            pairwise_distances = cdist(X, Z, metric='euclidean')

        sigma = np.median(pairwise_distances)
        gamma = 1 / (2 * (sigma ** 2)) if sigma != 0 else 1.0
        return gamma

    def _compute_kernels(self, X, gamma):
        #with parallel_config(backend='loky', inner_max_num_threads=1):
        Ks = Parallel(n_jobs=-1, use_threads=True)(
            delayed(self._compute_feature_kernel)(X[:, j].reshape(-1, 1), gamma) for j in range(self.d)
        )
        return Ks

    def _compute_feature_kernel(self, Xj, gamma):
        K = rbf_kernel(Xj, gamma=gamma)
        np.fill_diagonal(K, 0)
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

    def _compute_esp(self, kernel_matrices):
        s = [sum([np.power(k, p) for k in kernel_matrices]) for p in range(0, len(kernel_matrices) + 1)]
        e = [np.ones_like(kernel_matrices[0])]
        for r in range(1, len(kernel_matrices) + 1):
            term = 0
            for k in range(1, r + 1):
                term += ((-1) ** (k - 1)) * e[r - k] * s[k]
            e.append(term / r)
        return e

    def explain(self):
        shapley_values = np.zeros(self.d)
        
        onevec_X = np.ones_like(self.KXs[0])
        onevec_Z = np.ones_like(self.KZs[0])
        onevec_XZ = np.ones_like(self.KXZs[0])

        np.fill_diagonal(onevec_Z, 0)   
        np.fill_diagonal(onevec_X, 0)
        
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
            
            assert np.sum(np.diag(result_X)) == 0, "Diagonal elements should be zero."
            assert np.sum(np.diag(result_Z)) == 0, "Diagonal elements should be zero." 

            X_contribution = ( (self.n * (self.n-1) )**-1) * np.sum(result_X)
            Z_contribution = ( (self.m * (self.m-1) )**-1) * np.sum(result_Z)
            XZ_contribution =( (self.n * self.m )**-1) * np.sum(result_XZ)

            shapley_value_j = X_contribution + Z_contribution - 2 * XZ_contribution  
            
            return shapley_value_j.item()

        shapley_values = Parallel(n_jobs=-1)(
            delayed(shapley_value_j)(j) for j in range(self.d)
        )
        
        return shapley_values

# Example Usage
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    X = np.random.multivariate_normal(mean=[2, 3, 7, 1, 0], cov=np.eye(5), size=1000)
    Z = np.random.multivariate_normal(mean=[5, 3, 2, 5, 0], cov=np.eye(5), size=1000)

    explainer = MMDExplainer(X=X, Z=Z)
    sv = explainer.explain()
    print("MMD Shapley Values:", sv)
    print("Sum of Shapley values:", np.sum(sv))

    K_X = rbf_kernel(X, gamma = explainer.gamma_x)
    K_Z = rbf_kernel(Z, gamma = explainer.gamma_z)
    K_XZ = rbf_kernel(X, Z, gamma = explainer.gamma_xz)
    X_contribution = ((explainer.n * (explainer.n-1) )**-1) * ( np.sum(K_X) - np.sum(np.diag(K_X)) )
    Z_contribution = ((explainer.m * (explainer.m-1) )**-1) * ( np.sum(K_Z) - np.sum(np.diag(K_Z)) )
    XZ_contribution = 2 * np.mean(K_XZ)
    mmd_estimate =  X_contribution + Z_contribution - XZ_contribution 
