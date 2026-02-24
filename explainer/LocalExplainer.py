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
from .esp import ESPComputer

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
        

class RBFLocalExplainer(ProductKernelLocalExplainer):
    
    def __init__(self, model, value_function: str = "observational", esp_method: str = "quadratic", esp_kwargs: dict = None):
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
        # ESP method selection for internal ESPComputer usage (default: 'quadratic')
        self.esp_method = esp_method
        self.esp_kwargs = esp_kwargs or {}

        # Keep a copy of the raw model coefficients; we may reweight for observational mode.
        self.alpha_raw = self.alpha.copy()

        # Precompute nu_j^{(i)} = E_{X_j}[k_j(X_j, x_j^{(i)})] empirically from training set.
        # Shape: (n_samples, n_features). Used when value_function == 'observational'.
        self.nu = self._precompute_nu(self.X_train)

        # For observational value function, reweight alpha as:
        # 	ilde{alpha}_i = alpha_i * prod_j nu_j^{(i)}.
        if self.value_function == "observational":
            prod_nu = np.prod(self.nu, axis=1)
            self.alpha = self.alpha_raw * prod_nu
            # Null game under observational: v(empty) = sum_i alpha_i * prod_j nu_j^{(i)}
            self.null_game = float(np.sum(self.alpha))
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

        return np.array(kernel_vectors)

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
        
        kernel_vectors = self.compute_kernel_vectors(self.X_train, x).T        
        # Additional: compute Shapley via reusable ESPComputer for comparison
        start_time = time.time()
        esp = ESPComputer(method=self.esp_method, **self.esp_kwargs)
        Omega = esp.compute_weight_vectors(kernel_vectors)
        K = np.stack(kernel_vectors, axis=0)
        shapley_values = np.array([self.alpha.dot((K[:,j] - 1) * Omega[:,j]) for j in range(self.d)])
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
    X, y = make_regression(n_samples=1000, n_features=500, random_state=42)

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
