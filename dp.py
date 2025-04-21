
import numpy as np
import itertools
import scipy
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel


def instancewise_shapley_value(x, X_train, kernel_machine):
        X_sv = X_train[kernel_machine.support_,:]
        n, d = X_sv.shape
        
        gamma = kernel_machine._gamma
        kernel_mat = compute_kernel_matrix(x, X_sv, gamma)

        sv = np.zeros((d,1))

        for i in range(d):
            sv[i],_,_ = instancewise_sv_dim(kernel_mat, i, kernel_machine.dual_coef_.squeeze())

        return sv

def instancewise_sv_dim(Ks, dim, alpha):
    n, d = Ks.shape
    dp = np.zeros((d, d, n))
    

    Ks_copy = Ks.copy()
    Ks_copy[:, 0] = Ks[:, dim]
    Ks_copy[:, dim] = Ks[:, 0]

    sum_current = np.zeros((n,))

    # Fill the first order dp (base case)
    for j in range(d):
        dp[0, j, :] = Ks_copy[:, j]
        sum_current += Ks_copy[:, j]

    for i in range(1, d):
        temp_sum = np.zeros((n,))
        for j in range(d):
            # Subtract the previous contribution of this feature when moving to the next order
            sum_current -= dp[i - 1, j, :]

            dp[i, j, :] = Ks_copy[:, j] * sum_current
            temp_sum += dp[i, j, :]

        sum_current = temp_sum

    weights = np.zeros((d,1))
    # Loop over all possible subset sizes from 0 to n_features - 1
    for subset_size in range(d):
        weights[subset_size] = 1 / (scipy.special.comb(d - 1, subset_size) * d)

    dp[:,0,:] *= weights

    weights_p = np.zeros((d,1))
    weights_p[:-1] = -weights[1:]
    for i in range(1,d):
        dp[:,i,:] *= weights_p

    aggregate_k = np.sum(dp, axis=(0,1))
    sv = aggregate_k @ alpha
    return sv, aggregate_k, dp   

# Function to compute the RBF kernel matrix for each feature
def compute_kernel_matrix(x, X_train, gamma):
    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]

    # Initialize the kernel matrix
    kernel_matrix = np.zeros((n_samples, n_features))

    # For each sample and each feature, compute k(x_i^j, x^j)
    for i in range(n_samples):
        for j in range(n_features):
            kernel_matrix[i, j] = rbf_kernel(X_train[i, j].reshape(-1,1), x[j].reshape(-1,1), gamma=gamma)

    return kernel_matrix

# RBF kernel function
def rbf_kernel_manual(x1, x2, gamma):
    return np.exp(-gamma * np.sum((x1 - x2) ** 2))

# Function to compute v(S) for a subset of features S
def compute_v_S(X, alpha, X_train, gamma, S):
    v_S = 0
    for i in range(X_train.shape[0]):
        # Compute the product kernel for the features in subset S
        k_S = np.prod([rbf_kernel(X_train[i, s].reshape(-1,1), X[s].reshape(-1,1), gamma) for s in S])
        v_S += alpha[i] * k_S
    return v_S

# Function to compute Shapley values for each feature
def compute_shapley_values_bruteforce(X_instance, X_train, svm_model, gamma):
    n_features = X_train.shape[1]
    shapley_values = np.zeros(n_features)

    # Get dual coefficients (alpha) and support vectors from the trained SVM
    alpha = svm_model.dual_coef_[0]
    support_vectors = svm_model.support_vectors_

    # For each feature, compute the Shapley value
    for j in range(n_features):
        shapley_j = 0
        # Loop over all subsets of features excluding j
        for S in itertools.chain.from_iterable(itertools.combinations(range(n_features), r) for r in range(n_features)):
            if j not in S:
                S_with_j = S + (j,)
                v_S = compute_v_S(X_instance, alpha, support_vectors, gamma, S)
                v_S_with_j = compute_v_S(X_instance, alpha, support_vectors, gamma, S_with_j)
                shapley_j += (v_S_with_j - v_S) / (scipy.special.comb(n_features - 1, len(S)))

        shapley_values[j] = shapley_j / n_features

    return shapley_values


if __name__ == "__main__":
    '''
    Classification with SVM
    '''
    # Generate a synthetic dataset with 10 features
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train an SVM model with RBF kernel
    #svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
    #svm_model.fit(X_train, y_train)

    # Set gamma value for RBF kernel
    #gamma = svm_model._gamma

    '''
    Regression with SVR
    '''

    # Generate a synthetic regression dataset with 10 features
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train an SVR model with RBF kernel
    svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')
    svr_model.fit(X_train, y_train)

    # Set gamma value for RBF kernel
    gamma = svr_model._gamma

    # Select a test instance
    test_instance = X_test[0]

    # Compute Shapley values for the test instance
    #shapley_values = compute_shapley_values_bruteforce(test_instance, X_train, svr_model, gamma)

    # Print the computed Shapley values
    # print("Shapley values for the test instance:")
    # for i, value in enumerate(shapley_values):
    #     print(f"Feature {i+1}: {value:.4f}")

    sv = instancewise_shapley_value(test_instance, X_train, svr_model)

    # Compute the kernel matrix for a selected test instance
    kernel_matrix = compute_kernel_matrix(test_instance, X_train, gamma)
    dp = instancewise_sv_dim(kernel_matrix, 0)

    kernel_matrix.shape

