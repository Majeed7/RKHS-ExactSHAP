from itertools import combinations, chain
from random import seed
import random
from math import comb , factorial
import itertools
import copy

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.svm import SVR, LinearSVR
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array

import numpy as np
from scipy.linalg import cholesky, cho_solve
import scipy.special

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import lars_path, LassoLarsIC
import heapq

''' 
Game generation and subset generation/selection
'''

player_no_exact = 12
num_samples = 100000

# Generate a random game
def generate_game(N, q=None, essential=True):
    if q == None: q = N
    if ~essential: q = 1
    players = list(range(1, N + 1))

    subsets_all = all_subsets(players)
    if N <= player_no_exact:
        subset_values_mobius = {subset: np.random.uniform(-1, 1) / len(subset) if len(subset) <= q and len(subset) > 0 else 0 for subset in subsets_all }
        subset_values_mobius[frozenset({})] = 0
    else:
        sample_size = np.min((2**N, num_samples))
        subsets_all = random_subsets(players, sample_size)
        subset_values_mobius = {subset: np.random.uniform(-1, 1) / len(subset) if len(subset) <= q and len(subset) > 0 else 0 for subset in subsets_all }
        subset_values_mobius[frozenset({})] = 0
        
    subset_values = {}
    for set in subsets_all:
        subset_values[set] = sum([subset_values_mobius[x] for x in subset_values_mobius if x.issubset(set)])

    return players, subset_values_mobius, subsets_all, subset_values

# Function to generate random subsets
def random_subsets(full_set, num_samples=1000):
    """Generate a list of random subsets of the full set."""
    subsets = []
    for _ in range(num_samples):
        k = np.random.randint(0, len(full_set))
        subsets.append(frozenset(np.random.choice(full_set, k, replace=False)))
    return subsets

# Build all subsets
def all_subsets(full_set):
    """Generate all subsets of the full set."""
    subsets = []
    for r in range(len(full_set) + 1):
        for subset in combinations(full_set, r):
            subsets.append(frozenset(subset))
    return subsets

'''
Exact Shapley value Calculation 
'''

# Calculating exact Shpaley Value based on the generated game
def exact_Shapley_value(subset_values_mobius, N):
    shapley_values = np.zeros((N,))
    for i in range(N):
        shapley_values[i] = sum([subset_values_mobius[x] / len(x) for x in subset_values_mobius if frozenset({i+1}).issubset(x)])
    return shapley_values    

def interactions_from_alpha(matrix, values):
        interactions_dict = {}
        
        # Process each row in the matrix along with its corresponding value
        for row, value in zip(matrix, values):
            #if abs(value) < np.mean(np.abs(values)) and abs(value) < 1e-5: continue 
            if abs(value) < 1e-5: continue 

            n = len(row)
            indices = [i for i in range(n) if row[i] == 1]                
            
            
            # Generate all subsets for the indices where the row has 1s
            for size in range(2, len(indices) + 1):
                for combo in combinations(indices, size):
                    # Create a set representing the subset
                    subset_set = frozenset(combo)
                    
                    # Add the value to this subset key in the dictionary
                    if subset_set in interactions_dict:
                        interactions_dict[subset_set] += value
                    else:
                        interactions_dict[subset_set] = value

        return list(interactions_dict.items())
    
def gemfix_reg(X, y, sample_weight):
    n, d = X.shape
    inner_prod = np.inner(X,X) # X @ X.T
    lam = 0.0000000001 
    Omega = (2 ** inner_prod - 1) + lam * np.diag(sample_weight)

    sample_set_size = np.array(X @ np.ones((d,)), dtype=int)
    size_weight = np.zeros((d,))
    for i in range(1,d+1):
        for j in range(1,i+1):
            size_weight[i-1] += (1/j) * comb(i-1,j-1)
    
    alpha_weight = np.array([size_weight[t-1] if t != 0 else 0 for t in sample_set_size])
    
    L = cholesky(Omega , lower=True)
    alpha = cho_solve((L, True), y)

    shapley_val = np.zeros((d,))
    for i in range(d):
        #shapley_val[i] = (alpha_weight_sv * X_sv[:,i]) @ alpha
        shapley_val[i] = (alpha_weight * X[:,i]) @ alpha

    #print(f"the difference between the two shapley vlaue is {np.linalg.norm(shapley_val - shapley_val2, np.inf)}")    

    return shapley_val 

''' 
Estimating Shapley Value with Regression
'''
# Generate the weights for SHAP regression samples 
def generate_weights_by_size(total_players):
    weights_by_size = {}
    for size in range(total_players + 1):  # Include 0 to d
        if size > 0 and total_players - size > 0:  # Valid coalition sizes
            weight = (total_players - 1) / (comb(total_players, size) * size * (total_players - size))
            weights_by_size[size] = weight
        else:  # Handling for empty set and full set
            weights_by_size[size] = 100000  # Assign as needed, e.g., 0
    return weights_by_size

# Building data matrix based on the given subsets
def subset_to_matrix(N, subsets, subset_values):
    # Linear Regression 
    data_binaryfeature = []
    weights = []
    y = []
    weights_by_size = generate_weights_by_size(N)

    # Fill the matrix with binary representations
    for subset in subsets:
        row = [1 if i in subset else 0 for i in range(1,N+1)]
        weights.append(weights_by_size[len(subset)])
        y.append(subset_values[subset])
        data_binaryfeature.append(row)

    data_binaryfeature = np.array(data_binaryfeature)
    y = np.array(y)
    weights = np.array(weights)

    return data_binaryfeature, y, weights 

# Using regression for estimating SHAP values
def Shapley_regression(data_binaryfeature, y, weights, model_type='linear'):

    if model_type == 'linear':
        model = LinearRegression(fit_intercept=False)
        model.fit(data_binaryfeature, y, sample_weight=weights)
    
    elif model_type == 'shap_reg':
        #model = RidgeCV(cv=5, fit_intercept=False)
        model = Ridge(alpha=0.1)
        model.fit(data_binaryfeature, y, sample_weight=weights)

    return model

# Constructing the data matrix for regression based on the mobius transformation 
def subset_to_matrix_mobius(N, subsets, subsets_all, subset_values):
    matrix_size = len(subsets_all) #2 ** N 
    matrix_mobius = np.zeros((len(subsets), matrix_size)) #[[0 for _ in range(matrix_size)] for _ in range(matrix_size-1)]      
    weights = []
    y = []
    # Fill the matrix
    for row_idx, coalition in enumerate(subsets):
        weights.append(10000) if (len(coalition) == N  or len(coalition) == 0) else weights.append(1)
        y.append(subset_values[coalition])
        for col_idx, subset in enumerate(subsets_all):
            # If the subset is a subset of the coalition, mark as 1
            if subset.issubset(coalition):
                matrix_mobius[row_idx][col_idx] = 1

    return matrix_mobius, np.array(y), np.array(weights)

# Estimating the Mobius transformation of game values based on linear regression
def gemfix_regression(matrix_mobius, y, weights, subsets_all, N, model_type='linear', alpha=1.0):

    if model_type == 'linear':
        model_gemfix = LinearRegression(fit_intercept=True)

    elif model_type == 'ridge':
        #model_mobius = Ridge(alpha=alpha, fit_intercept=False)
        model_gemfix = RidgeCV(cv=5, fit_intercept=True)

    elif model_type == 'lasso':
        model_gemfix = Lasso(alpha=alpha, fit_intercept=True)

    elif model_type == 'lassocv':
        model_gemfix = LassoCV(cv=5, random_state=0, fit_intercept=True)

    elif model_type == 'svr':
        model_gemfix = LinearSVR(fit_intercept=True)

    elif model_type == 'gemfix_reg':
        shapley_values, model = gemfix_reg(matrix_mobius, y, weights)
        return shapley_values, model

    model_gemfix.fit(matrix_mobius, y, sample_weight=weights)
    gemfix_shapely = gemfix_shapley_calculation(subsets_all, model_gemfix.coef_, N)

    return np.array(gemfix_shapely), model_gemfix

# Computing Shapley value based on the Mobius transformation values
def gemfix_shapley_calculation(subsets_all, coef, N):
    shapley_mobius = []
    for i in range(1,N+1):
        subsets_indices = [(index) for index, subset in enumerate(subsets_all) if i in subset]
        subsets_weight = [(1 / len(subset)) for _, subset in enumerate(subsets_all) if i in subset]
        shapley_mobius.append(np.sum(coef.squeeze()[subsets_indices] * subsets_weight))

    return shapley_mobius


'''
Sampling Strategy as in KernelSHAP
'''

def kernel_shap_sampling(M, nsamples):
    """
    Perform Kernel SHAP sampling to generate a list of unique samples.
    Each sample is a tuple (mask, weight) where:
      - mask is a binary numpy array of length M indicating feature presence.
      - weight is the kernel weight assigned to that sample.
    
    Parameters:
      M        : int, total number of features.
      nsamples : int, total number of samples to generate.
    
    Returns:
      List of tuples (mask, weight)
    """
    samples = []       # List to store (mask, weight) tuples
    kernelWeights = [] # List to store weights (for normalization updates)

    # Step 1: Weight initialization and scaling
    num_subset_sizes = int(np.ceil((M - 1) / 2.0))
    num_paired_subset_sizes = int(np.floor((M - 1) / 2.0))
    weight_vector = np.array([(M - 1.0) / (i * (M - i)) for i in range(1, num_subset_sizes + 1)])
    # For the first half (paired subsets) double the weights
    if num_paired_subset_sizes > 0:
        weight_vector[:num_paired_subset_sizes] *= 2
    weight_vector /= np.sum(weight_vector)


    num_full_subsets = 0
    num_samples_left = nsamples
    group_inds = np.arange(M, dtype=int)
    mask = np.zeros(M, dtype=float)
    remaining_weight_vector = copy.copy(weight_vector)

    # Helper to add sample (mask copy and weight) to our lists
    def add_sample(mask_array, weight_val):
        samples.append(mask_array.copy())
        kernelWeights.append(weight_val)

    # Step 2: Fixed enumeration of subsets for lower subset sizes
    for subset_size in range(1, num_subset_sizes + 1):
        # Calculate total number of subsets for this size
        nsubsets = scipy.special.comb(M, subset_size, exact=True)
        if subset_size <= num_paired_subset_sizes:
            nsubsets *= 2
        
        if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
            num_full_subsets += 1
            num_samples_left -= nsubsets

        if remaining_weight_vector[subset_size - 1] < 1.0:
            remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])
        
        # Weight for each fixed enumeration sample for the current subset size
        w = weight_vector[subset_size - 1] / scipy.special.comb(M, subset_size, exact=True)
        if subset_size <= num_paired_subset_sizes:
            w /= 2.0
        
        # Enumerate all combinations for the current subset size
        for inds in itertools.combinations(group_inds, subset_size):
            mask.fill(0.0)
            for i in inds:
                mask[i] = 1.0
            add_sample(mask, w)
            # If the current subset has a complement sample, add it too.
            if subset_size <= num_paired_subset_sizes:
                comp_mask = np.abs(mask - 1)
                add_sample(comp_mask, w)
        # If the current subset size exceeds what we can fully enumerate, exit loop.
        if subset_size > num_full_subsets:
            break

    nfixed_samples = len(samples)
    samples_left = nsamples - nfixed_samples

    # Step 3: Random sampling from the remaining subset space
    if num_full_subsets != num_subset_sizes and samples_left > 0:
        # Adjust remaining weight vector: consider remaining sizes
        temp_remaining_weight = copy.copy(weight_vector)
        if num_paired_subset_sizes > 0:
            temp_remaining_weight[:num_paired_subset_sizes] /= 2.0  # each sample yields two outputs
        remaining_weight = temp_remaining_weight[num_full_subsets:]
        if np.sum(remaining_weight) > 0:
            remaining_weight /= np.sum(remaining_weight)
        else:
            remaining_weight = np.ones_like(remaining_weight) / len(remaining_weight)
        
        # Draw size indices according to the remaining weight vector
        ind_set = np.random.choice(len(remaining_weight), 4 * samples_left, p=remaining_weight, replace=True)
        ind_set_pos = 0
        used_masks = {}
        while samples_left > 0 and ind_set_pos < len(ind_set):
            mask.fill(0.0)
            ind = ind_set[ind_set_pos]
            ind_set_pos += 1
            # Compute subset_size from drawn index
            subset_size = ind + num_full_subsets + 1
            # Randomly select subset indices
            perm = np.random.permutation(M)
            mask.fill(0.0)
            mask[perm[:subset_size]] = 1.0

            mask_tuple = tuple(mask.tolist())
            new_sample = False
            if mask_tuple not in used_masks:
                new_sample = True
                used_masks[mask_tuple] = len(samples)
                add_sample(mask, 1.0)
                samples_left -= 1
            else:
                # If this mask was seen, increment its weight
                idx = used_masks[mask_tuple]
                kernelWeights[idx] += 1.0

            # Add complement sample if applicable
            if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                comp_mask = np.abs(mask - 1)
                comp_tuple = tuple(comp_mask.tolist())
                if new_sample:
                    used_masks[comp_tuple] = len(samples)
                    add_sample(comp_mask, 1.0)
                    samples_left -= 1
                else:
                    if comp_tuple in used_masks:
                        idx = used_masks[comp_tuple]
                        kernelWeights[idx] += 1.0

        # Normalize the kernel weights for the random samples to equal the weight remaining
        weight_left = np.sum(weight_vector[num_full_subsets:])
        random_indices = list(range(nfixed_samples, len(kernelWeights)))
        random_total = np.sum([kernelWeights[i] for i in random_indices])
        if random_total > 0:
            factor = weight_left / random_total
            for i in random_indices:
                kernelWeights[i] *= factor

    # Assemble final list of samples with their weights
    final_samples = [(mask, weight) for mask, weight in zip(samples, kernelWeights)]
    return final_samples


# Function to generate feature subsets
def generate_feature_subsets(n_features, n_samples):
    subsets = []  # Initialize empty list to store subsets

    # Loop to generate the required number of subsets
    for _ in range(n_samples):
        # Determine subset size based on weighted sampling
        subset_size = np.random.choice(
            range(n_features + 1), 
            p=shapley_kernel_weights(n_features)
        )

        # Randomly select 'subset_size' number of features
        subset_indices = np.random.choice(
            range(n_features), 
            subset_size, 
            replace=False
        )

        # Create subset binary vector
        subset = np.zeros(n_features)
        subset[subset_indices] = 1
        
        # Store the resulting subset
        subsets.append(subset)

    return subsets  # Return the list of feature subsets

# Function to calculate Shapley kernel weights
def shapley_kernel_weights(n_features):
    weights = []
    for k in range(n_features + 1):
        # Compute the weight according to Shapley kernel formula
        weight = (n_features - 1) / (scipy.special.comb(n_features, k) * k * (n_features - k))
        weights.append(weight)

    weights = np.array(weights)
    weights[np.isinf(weights)] = 0  # Handle divide by zero for k=0 or N
    return weights / weights.sum()  # Normalize weights





if __name__ == '__main__':
    '''
    testing Sampling 
    '''

    # Example usage
    n_features = 10  # Total number of features
    n_samples = 1000  # Number of subsets to generate
    feature_subsets = generate_feature_subsets(n_features, n_samples)



    '''
    Testing the methods
    '''
    seed(42)  # For reproducibility

    N = 10 # number of players
    players, subsets_all, subset_values, subset_values_mobius = generate_game(N, 3) # generate a game

    exact_shapley_values = exact_Shapley_value(subsets_all, subset_values, N, players) # computing the exact Shapely value of the game
    print(f"Exact shapley value is: {np.round(exact_shapley_values, 3)}")


    # Subset selection for SV estimation
    num_samples = 200
    drawn_samples = np.min((num_samples, 2 ** N))
    #subsets_all.pop(0)
    subsets = random.sample(subsets_all, drawn_samples) # subsets_all #
    # We need to have the game of all players in the regression analysis just to make sure the sum of Shapley values of features is the predicted value
    if subsets_all[-1] not in subsets: subsets.append(subsets_all[-1])
    if subsets_all[0] not in subsets: subsets.append(subsets_all[0])

    # generate data matrix for linear regression to estimate SHAP value
    data_binaryfeature, y, weights = subset_to_matrix(N, subsets, subset_values)
    model = Shapley_regression(data_binaryfeature, y, weights)
    shap_values = model.coef_
    print(f"SHAP regression values: {np.round(shap_values, 3)}")

    #model_rr = Shapley_regression(data_binaryfeature, y, weights, model_type='ridge')
    #shap_values_rr = model_rr.coef_
    #print(f"SHAP regression values: {np.round(shap_values_rr, 3)}")


    # generate data matrix for linear regression estiamting Mobius transformation of game values
    matrix_mobius, y_mobius, weights = subset_to_matrix_mobius(N, subsets, subsets_all, subset_values)

    shapley_mobius_ksvr, ksvr_model = Shapley_mobius_regression(data_binaryfeature, y, weights, subsets_all, N, model_type='ksvr')
    print(f"KSVR   Shapely value is {np.round(shapley_mobius_ksvr, 3)}")

    shapley_mobius_linear, linear_model = Shapley_mobius_regression(matrix_mobius, y_mobius, weights, subsets_all, N)
    print(f"Linear Shapely value is {np.round(shapley_mobius_linear, 3)}")

    shapley_mobius_ridge, ridge_model = Shapley_mobius_regression(matrix_mobius, y_mobius, weights, subsets_all, N, model_type='ridge')
    print(f"Ridge  Shapely value is {np.round(shapley_mobius_ridge, 3)}")

#    shapley_mobius_lassocv, lassocv_model = Shapley_mobius_regression(matrix_mobius, y_mobius, weights, subsets_all, N, model_type='lassocv')
#    print(f"Lasso    Shapely value is {shapley_mobius_lassocv}")

    shapley_mobius_svr, svr_model = Shapley_mobius_regression(matrix_mobius, y_mobius, weights, subsets_all, N, model_type='svr')
    print(f"SVR    Shapely value is {np.round(shapley_mobius_svr, 3)}")
    

    print("done!")









