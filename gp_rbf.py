import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.linalg import solve_triangular


def train_gp(X, y):
    """
    Train a Gaussian Process model for the given input X and y.
    The RBF kernel is adapted to the number of features in X.
    """
    # Define kernel with individual length scales for each feature and noise term
    n_features = X.shape[1]
    kernel = ConstantKernel() * RBF(length_scale=np.ones(n_features)) + WhiteKernel(noise_level=0.1)
    
    # Initialize and train GP
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=1, normalize_y=True)
    gp.fit(X, y)
    
    return gp

def compute_feature_wise_rbf_kernel(X_test, X_train, length_scales, constant_value=1.0):
    """
    Compute the feature-wise RBF kernel for a set of test samples using the RBF function from GP.
    Each feature-wise kernel is multiplied by the n_feature root of the constant value.
    """
    n_features = X_test.shape[1]
    feature_kernels = []
    constant_root = constant_value ** (1 / n_features)
    
    for i in range(n_features):
        # Use RBF kernel for each feature independently
        rbf_kernel = RBF(length_scale=length_scales[i])
        kernel = constant_root * rbf_kernel(X_test[:, i:i+1], X_train[:, i:i+1])
        feature_kernels.append(kernel)
    
    return feature_kernels


# Generate synthetic data
np.random.seed(0)
X_train = np.random.rand(20, 2)  # 20 samples, 2 features
y_train = np.sin(X_train[:, 0] + X_train[:, 1])

# Define kernel with individual length scales for each feature and noise term
kernel = ConstantKernel() * RBF(length_scale=[1.0, 1.0]) + WhiteKernel(noise_level=0.1)

# Initialize and train GP
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.)
gp.fit(X_train, y_train)

# Extract optimized kernel parameters
constant_value = gp.kernel_.k1.k1.constant_value
rbf_length_scales = gp.kernel_.k1.k2.length_scale
noise_level = gp.kernel_.k2.noise_level

def custom_gp_predict(X_test):
    """
    Custom GP prediction function that returns mean and standard deviation
    """
    # RBF kernel between test and training points
    K_star = constant_value * RBF(length_scale=rbf_length_scales)(X_test, gp.X_train_)
    
    # Mean prediction
    mean = K_star @ gp.alpha_
    
    # Variance prediction
    K_star_star = RBF(length_scale=rbf_length_scales)(X_test)
    L = gp.L_  # Cholesky decomposition of K(X_train, X_train)
    
    # Compute variance using Cholesky decomposition
    V = solve_triangular(L, K_star.T, lower=True)
    var = K_star_star.diagonal() - np.sum(V**2, axis=0)
    
    # Add noise variance if predicting noisy values
    # var += noise_level
    
    return mean, np.sqrt(np.abs(var))  # Ensure non-negative standard deviation

# Example usage
X_test = np.random.rand(5, 2)
mean_pred, std_pred = custom_gp_predict(X_test)

print("Predicted means:", mean_pred)
print("Predicted stds:", std_pred)

# Verify against sklearn's implementation
mean_sklearn, std_sklearn = gp.predict(X_test, return_std=True)
print("\nSklearn means:", mean_sklearn)
print("Sklearn stds:", std_sklearn)

