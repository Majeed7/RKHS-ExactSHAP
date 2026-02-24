import numpy as np
import math
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist, cdist
from joblib import Parallel, delayed, parallel_config

try: 
    from esp import ESPComputer
except ImportError:
    from .esp import ESPComputer
import seaborn as sns

class MMDExplainer:
    def __init__(self, X, Z, **kwargs):
        """
        Initialize the MMD Explainer using keyword arguments.
        
        Args:
            X: Input feature matrix of shape (n_samples, n_features).
            Z: Comparison feature matrix of shape (m_samples, n_features).
            kernel_x: Kernel type for features ('rbf' supported). Default 'rbf'.
            gamma: Bandwidth parameter for feature RBF kernels. If None, use median heuristic.
        """
        self.X = X
        self.Z = Z
        self.n, self.d = self.X.shape
        self.m = self.Z.shape[0]
        
        self.estimation_type = kwargs.get("estimation_type", "U")

        # Value function selection
        # - "observational" (default): interventional value function with empirical expectations
        #   c_j^P, c_j^Q, c_j^{PQ} and C_{-j} factors (as in the paper).
        # - "baseline": legacy mode matching the earlier implementation (baseline = 1).
        self.value_function = kwargs.get("value_function", "observational")

        # ESP implementation selection (uses explainer/esp.py)
        self.esp_method = kwargs.get("esp_method", "quadratic")
        self.esp_kwargs = kwargs.get("esp_kwargs", {}) or {}
        self._esp = ESPComputer(method=self.esp_method, **self.esp_kwargs)
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

        if self.value_function not in {"observational", "baseline"}:
            raise ValueError(f"Unknown value_function: {self.value_function}")

        # Empirical constants for observational/interventional value function.
        # These are used only when value_function == "observational".
        if self.value_function == "observational":
            self._precompute_c_constants()

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

    def _pair_mean(self, K: np.ndarray, estimation_type: str) -> float:
        """Empirical expectation of k(X,X') under the chosen estimator.

        For U-statistic we average off-diagonal terms; for V-statistic we average all terms.
        """
        if estimation_type == "U":
            n = K.shape[0]
            return float((np.sum(K) - float(np.trace(K))) / float(n * (n - 1)))
        if estimation_type == "V":
            n = K.shape[0]
            return float(np.mean(K))  # n^{-2} sum_{i,j}
        raise ValueError(f"Unknown estimation_type: {estimation_type}")

    def _precompute_c_constants(self) -> None:
        """Precompute c_j and C_{-j} factors for the observational/interventional value function."""
        # c_j^P, c_j^Q depend on within-sample estimator (U or V).
        self.cP = np.array([self._pair_mean(K, self.estimation_type) for K in self.KXs], dtype=np.float64)
        self.cQ = np.array([self._pair_mean(K, self.estimation_type) for K in self.KZs], dtype=np.float64)
        # Cross expectation always averages over all n*m pairs.
        self.cPQ = np.array([float(np.mean(K)) for K in self.KXZs], dtype=np.float64)

        eps = 1e-300  # RBF kernels are strictly positive; this is just a guard.
        self.CP_total = float(np.prod(np.maximum(self.cP, eps)))
        self.CQ_total = float(np.prod(np.maximum(self.cQ, eps)))
        self.CPQ_total = float(np.prod(np.maximum(self.cPQ, eps)))

        self.CminusP = (self.CP_total / np.maximum(self.cP, eps)).astype(np.float64)
        self.CminusQ = (self.CQ_total / np.maximum(self.cQ, eps)).astype(np.float64)
        self.CminusPQ = (self.CPQ_total / np.maximum(self.cPQ, eps)).astype(np.float64)


        self.cP = self.cPQ
        self.cQ = self.cPQ
        self.cP_total = self.CPQ_total
        self.cQ_total = self.CPQ_total
        self.CminusP = self.CminusPQ
        self.CminusQ = self.CminusPQ

        # Baseline (null coalition) value v(∅) for the observational/interventional game.
        # For an empty set S, k_S ≡ 1, so the empirical U/V averages are 1.
        self.null_game = float(self.CP_total + self.CQ_total - 2.0 * self.CPQ_total)
    
    def _gather_pairwise_feature_values(self, Ks, tri_i, tri_j):
        """Build a (num_pairs, d) array of per-feature kernel values for given index pairs."""
        # Stack features as last axis for ESPComputer: shape (num_pairs, d)
        return np.stack([K[tri_i, tri_j] for K in Ks], axis=1).astype(np.float64)

    def explain(self):
        # Legacy implementation (baseline = 1) kept for backward compatibility.
        if self.value_function == "baseline":
            triX_i, triX_j = np.triu_indices(self.n, k=1)
            triZ_i, triZ_j = np.triu_indices(self.m, k=1)

            Kpairs_X = self._gather_pairwise_feature_values(self.KXs, triX_i, triX_j)
            Kpairs_Z = self._gather_pairwise_feature_values(self.KZs, triZ_i, triZ_j)
            Kpairs_XZ = np.stack([K.reshape(-1) for K in self.KXZs], axis=1).astype(np.float64)

            Omega_X = self._esp.compute_weight_vectors(Kpairs_X)
            Omega_Z = self._esp.compute_weight_vectors(Kpairs_Z)
            Omega_XZ = self._esp.compute_weight_vectors(Kpairs_XZ)

            denom_x = float(self.coef_x)
            denom_z = float(self.coef_z)
            denom_xz = float(self.coef_xz)

            shapley = np.zeros(self.d, dtype=np.float64)
            for q in range(self.d):
                fX = (Kpairs_X[:, q] - 1.0) * Omega_X[:, q]
                fZ = (Kpairs_Z[:, q] - 1.0) * Omega_Z[:, q]
                sumX = 2.0 * float(np.sum(fX))
                sumZ = 2.0 * float(np.sum(fZ))
                # For V-statistic, diagonal contribution is always zero since (k-1)=0.
                X_contribution = sumX / denom_x
                Z_contribution = sumZ / denom_z

                fXZ = (Kpairs_XZ[:, q] - 1.0) * Omega_XZ[:, q]
                XZ_contribution = float(np.sum(fXZ)) / denom_xz
                shapley[q] = X_contribution + Z_contribution - 2.0 * XZ_contribution

            return shapley

        # ------------------------------------------------------------
        # Observational value function (paper):
        # Uses c_j and C_{-j} factors, and (k_j - c_j) rather than (k_j - 1).
        # Uses ESPComputer on per-pair kernel values (treating each pair as a "sample").
        # Exploits symmetry: sums over i!=j computed from upper triangle.
        # ------------------------------------------------------------

        # Upper-triangle (off-diagonal) indices for symmetric within-sample sums.
        triX_i, triX_j = np.triu_indices(self.n, k=1)
        triZ_i, triZ_j = np.triu_indices(self.m, k=1)

        # Per-pair kernel values: (num_pairs, d)
        Kpairs_X = self._gather_pairwise_feature_values(self.KXs, triX_i, triX_j)
        Kpairs_Z = self._gather_pairwise_feature_values(self.KZs, triZ_i, triZ_j)
        Kpairs_XZ = np.stack([K.reshape(-1) for K in self.KXZs], axis=1).astype(np.float64)

        # IMPORTANT: In the observational/interventional value function,
        # the complement-product factor C_{D\(S∪{q})} varies with S. This yields
        # ESPs over normalized feature-kernels k_l / c_l for l != q.
        # We normalize all feature columns; feature q is excluded inside omega anyway.
        Kpairs_X_tilde = Kpairs_X / self.cP[None, :]
        Kpairs_Z_tilde = Kpairs_Z / self.cQ[None, :]
        Kpairs_XZ_tilde = Kpairs_XZ / self.cPQ[None, :]

        # ESP weights omega per pair and feature: shape (num_pairs, d)
        Omega_X = self._esp.compute_weight_vectors(Kpairs_X_tilde)
        Omega_Z = self._esp.compute_weight_vectors(Kpairs_Z_tilde)
        Omega_XZ = self._esp.compute_weight_vectors(Kpairs_XZ_tilde)

        # For V-statistic, add diagonal contributions separately.
        if self.estimation_type == "V":
            omega_diag_P = self._esp.compute_weight_vectors((np.ones((1, self.d), dtype=np.float64) / self.cP[None, :]))[0]
            omega_diag_Q = self._esp.compute_weight_vectors((np.ones((1, self.d), dtype=np.float64) / self.cQ[None, :]))[0]
        else:
            omega_diag_P = None
            omega_diag_Q = None

        # Compute Shapley values feature-wise.
        shapley = np.zeros(self.d, dtype=np.float64)

        # Precompute denominators.
        denom_x = float(self.coef_x)
        denom_z = float(self.coef_z)
        denom_xz = float(self.coef_xz)

        for q in range(self.d):
            # Within-X term (symmetric): sum_{i!=j} = 2 * sum_{i<j}
            fX_off = (Kpairs_X_tilde[:, q] - 1.0) * Omega_X[:, q]
            sumX = 2.0 * float(np.sum(fX_off))
            if self.estimation_type == "V":
                # Add diagonal i=i (kernel is 1 for RBF): contributes n terms.
                sumX += float(self.n) * (1.0 - float(self.cP[q])) * float(omega_diag_P[q])
            X_contribution = float(self.CP_total) * (sumX / denom_x)

            # Within-Z term
            fZ_off = (Kpairs_Z_tilde[:, q] - 1.0) * Omega_Z[:, q]
            sumZ = 2.0 * float(np.sum(fZ_off))
            if self.estimation_type == "V":
                sumZ += float(self.m) * (1.0 - float(self.cQ[q])) * float(omega_diag_Q[q])
            Z_contribution = float(self.CQ_total) * (sumZ / denom_z)

            # Cross term (no symmetry in general): sum_{i,j}
            fXZ = (Kpairs_XZ_tilde[:, q] - 1.0) * Omega_XZ[:, q]
            XZ_contribution = float(self.CPQ_total) * (float(np.sum(fXZ)) / denom_xz)

            shapley[q] = X_contribution + Z_contribution - 2.0 * XZ_contribution

        return shapley

# Example Usage
if __name__ == "__main__":
    
    np.random.seed(42)
    d = 5
    n = 500
    X = np.random.multivariate_normal(mean=[8, 3, 7, 5, 1], cov=np.eye(5), size=3000)
    Z = np.random.multivariate_normal(mean=[5, 3, 2, 5, 6], cov=np.eye(5), size=3000)
    explainer = MMDExplainer(X=X, Z=Z, estiamtion_type="V")
    sv = explainer.explain()
    print("MMD Shapley Values:", sv)
    print("Sum of Shapley values:", np.sum(sv))
    import matplotlib.pyplot as plt

    num_seeds = 50
    all_svs = []
    all_sv_base = []
    for seed in range(num_seeds):
        rng = np.random.RandomState(seed)
        X = np.random.multivariate_normal(mean=[8, 3, 7, 5, 1], cov=np.eye(5), size=1000)
        Z = np.random.multivariate_normal(mean=[-5, 3, 2, 5, 6], cov=np.eye(5), size=1000)

        expl = MMDExplainer(X=X, Z=Z)
        sv = expl.explain()
        base = expl.cP + expl.cQ - 2 * expl.cPQ
        all_sv_base.append(sv+base)
        all_svs.append(sv)

    all_svs = np.vstack(all_svs)  # shape (num_seeds, d)
    all_sv_base = np.vstack(all_sv_base)
    d = all_svs.shape[1]

    fig, axes = plt.subplots(2, d, figsize=(3 * d, 6), squeeze=False)
    for q in range(d):
        sns.kdeplot(all_svs[:, q], ax=axes[0, q], fill=True)
        axes[0, q].set_title(f"Feature {q}")
        axes[0, q].set_xlabel("Shapley value")
        axes[0, q].grid(False)
    
    for q in range(d):
        sns.kdeplot(all_sv_base[:, q], ax=axes[1, q], fill=True)
        axes[1, q].set_title(f"Feature {q} (base adjusted)")
        axes[1, q].set_xlabel("Shapley value + base")
        axes[1, q].grid(False)

    plt.tight_layout()
    plt.show()

    K_X = rbf_kernel(X, gamma = explainer.gamma)
    K_Z = rbf_kernel(Z, gamma = explainer.gamma)
    K_XZ = rbf_kernel(X, Z, gamma = explainer.gamma)
    X_contribution = ((explainer.n * (explainer.n-1) )**-1) * ( np.sum(K_X) - np.sum(np.diag(K_X)) )
    Z_contribution = ((explainer.m * (explainer.m-1) )**-1) * ( np.sum(K_Z) - np.sum(np.diag(K_Z)) )
    XZ_contribution = 2 * np.mean(K_XZ)
    mmd_estimate =  X_contribution + Z_contribution - XZ_contribution 
