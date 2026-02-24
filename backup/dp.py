
import numpy as np
import itertools
import scipy
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
import time 
import sys
import threading

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
    dp = np.zeros((d, d, n), dtype=np.float64)
    

    Ks_copy = Ks.copy()
    Ks_copy[:, 0] = Ks[:, dim]
    Ks_copy[:, dim] = Ks[:, 0]
    sum_current = np.zeros((n,), dtype=np.float64)

    # Fill the first order dp (base case)
    for j in range(d):
        dp[0, j, :] = Ks_copy[:, j]
        sum_current += Ks_copy[:, j]

    for i in range(1, d):
        temp_sum = np.zeros((n,), dtype=np.float64)
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
    simulation = True
    if simulation == True:

        n_samples = 1000
        feature_list = [500, 1000] #[8,10, 12, 13, 15, 18, 20, 30, 50, 100, 200, 500, 1000]
        np.random.seed(42)

        for n_features in feature_list:
            times = []
            # Simulate a kernel matrix and alpha coefficients
            kernel_matrix_sim = np.random.rand(n_samples, n_features)
            alpha_sim = np.random.randn(n_samples)
            sv_all = []
            for j in range(n_features):
                start = time.time()
                sv_sim, _, _ = instancewise_sv_dim(kernel_matrix_sim, j, alpha_sim)
                end = time.time()
                times.append(end - start)
                sv_all.append(sv_sim)
            print(f"Features: {n_features} | time: {np.sum(times):.4f}s")
            import matplotlib.pyplot as plt

            # Prepare data for plotting
            features = [18, 20, 30, 50, 100, 200, 500]
            brute_force_times_raw = ["4.34", "45.55", "217.58", "380", ">300", ">300", ">300"]
            pkex_times = [0.022, 0.028, 0.094, 1.361, 18.258, 25.244, 80.244]

            # Only plot brute-force times that are not '>300'
            brute_force_features = [f for f, t in zip(features, brute_force_times_raw) if t[0].isdigit()]
            brute_force_times = [float(t) for t in brute_force_times_raw if t[0].isdigit()]

            plt.figure(figsize=(5, 4))
            plt.plot(brute_force_features, brute_force_times, marker='o', label='Brute-force', linewidth=4, markersize=10)
            plt.plot(features, pkex_times, marker='s', label='PKeX-Shapley', linewidth=4, markersize=10)

            plt.xlabel('Number of Features', fontsize=18, fontweight='bold')
            plt.ylabel('Time (s)', fontsize=18, fontweight='bold')
            # plt.title('Computation Time vs Number of Features', fontsize=18, fontweight='bold')
            plt.legend(fontsize=18)
            plt.grid(True)
            plt.tight_layout()

            plt.xticks(fontsize=16, fontweight='bold')
            plt.yticks(fontsize=16, fontweight='bold')
            plt.ylim(0, 300)

            plt.savefig(f"computation_time_vs_features.png", dpi=300)
            plt.show()
            continue 
            # Brute-force Shapley value computation using kernel matrix
            def compute_shapley_values_bruteforce_kernel(kernel_matrix, alpha):
                n_samples, n_features = kernel_matrix.shape
                shapley_values = np.zeros(n_features)
                # For each feature
                for j in range(n_features):
                    shapley_j = 0
                    # Loop over all subsets of features excluding j
                    for S in itertools.chain.from_iterable(itertools.combinations(range(n_features), r) for r in range(n_features)):
                        if j not in S:
                            S_with_j = S + (j,)
                            # v_S: product of kernel values for subset S, summed over samples and weighted by alpha
                            k_S = np.prod(kernel_matrix[:, S], axis=1) if S else np.ones(n_samples)
                            v_S = np.sum(alpha * k_S)
                            k_Sj = np.prod(kernel_matrix[:, S_with_j], axis=1)
                            v_S_with_j = np.sum(alpha * k_Sj)
                            shapley_j += (v_S_with_j - v_S) / (scipy.special.comb(n_features - 1, len(S)))
                    shapley_values[j] = shapley_j / n_features
                return shapley_values
            
            
            def run_with_timeout(func, args=(), kwargs={}, timeout=10):
                result = [None]
                def target():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        result[0] = e
                thread = threading.Thread(target=target)
                thread.start()
                thread.join(timeout)
                if thread.is_alive():
                    print(f"Brute-force Shapley computation exceeded {timeout} seconds, skipping.")
                    return None
                return result[0]
            
            start_bruteforce = time.time()
            shapley_bruteforce = run_with_timeout(compute_shapley_values_bruteforce_kernel, args=(kernel_matrix_sim, alpha_sim), timeout=300)
            end_bruteforce = time.time()
            print(f"Brute-force Shapley computation time: {end_bruteforce - start_bruteforce:.4f}s")

        sys.exit()

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


