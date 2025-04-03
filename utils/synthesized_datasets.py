import numpy as np
import random
import matplotlib.pyplot as plt



data_names=['Sine Log', 'Sine Cosine', 'Poly Sine', 'Squared Exponentials', 'Tanh Sine', 
            'Trigonometric Exponential', 'Exponential Hyperbolic']

#def generate_X(n_samples=100, n_features=10):
#    import numpy as np

def generate_X(num_samples, num_features, influential_indices, correlation_value=0.6, type="correlated"):

    if type != "correlated":
    # Generate samples with a standard normal distribution
        return np.random.randn(num_samples, num_features)
    """
    Generate synthetic data where specified influential features are correlated with unique non-influential features.

    Parameters:
    - num_samples: int, number of samples.
    - num_features: int, total number of features.
    - influential_indices: list of int, indices of influential features.
    - correlation_value: float, correlation value between paired features.

    Returns:
    - data: np.ndarray, generated synthetic data.
    """
    num_influential = len(influential_indices)
    non_influential_indices = list(set(range(num_features)) - set(influential_indices))

    assert num_influential <= len(non_influential_indices), (
        "Number of non-influential features must be >= number of influential features."
    )
    
    # Initialize covariance matrix (identity matrix)
    cov_matrix = np.eye(num_features)
    
    # Assign correlations between each influential feature and a unique non-influential feature
    for i, inf_idx in enumerate(influential_indices):
        non_inf_idx = non_influential_indices[i]
        cov_matrix[inf_idx, non_inf_idx] = correlation_value
        cov_matrix[non_inf_idx, inf_idx] = correlation_value  # Symmetry
    
    # Check positive definiteness of the covariance matrix
    if not np.all(np.linalg.eigvals(cov_matrix) > 0):
        raise ValueError("Covariance matrix is not positive definite. Adjust correlations or feature setup.")
    
    # Generate mean vector
    mean = np.zeros(num_features)
    
    # Generate data with correlation
    data = np.random.multivariate_normal(mean, cov_matrix, size=num_samples)
    
    return data

def generate_dataset_sinlog(n_samples=100, n_features=10, type="correlated"):
    influential_indices = np.arange(0, 2) 
    X = generate_X(n_samples, n_features, influential_indices = influential_indices, correlation_value=0.6, type=type)

    def fn(X):
        f1, f2 = X[:, 0], X[:, 1]

        # Main effects
        main_effect_1 = np.sin(f1)  # Main effect from feature 1
        main_effect_2 = np.log1p(np.abs(f2))  # Main effect from feature 2

        # Complex interaction effect between feature 1 and feature 2
        interaction_effect = np.sin(f1 * f2) + np.exp(-((f1 - f2) ** 2))

        # Combine main effects and interaction effect
        y = main_effect_1 + main_effect_2 + interaction_effect
        return y

    y = fn(X)
    
    return X, y, fn, influential_indices, 'Sine Log'

def generate_dataset_sin(n_samples=100, n_features=10, noise=0.1, type="correlated"):
    """
    Args:
        noise (float): Standard deviation of Gaussian noise to add to the output. 
    """
    influential_indices = np.arange(0, 2)
    X = generate_X(n_samples, n_features, influential_indices=influential_indices, correlation_value=0.6, type=type)

    def fn(X):
        f1, f2 = X[:, 0], X[:, 1]

        # Main effects: functions of each individual feature
        main_effects = np.sin(f1) + 0.5 * np.cos(f2)
        
        # Interaction term: product of two features (interaction between features 1 and 2)
        interaction = f1 * f2
        
        # Combine main effects and interaction to compute the true target values
        y_true = main_effects + interaction
        
        # Add Gaussian noise to the target values
        #noise_array = noise * np.random.randn(X.shape[0])
        y = y_true #+ noise_array
        
        return y

    y = fn(X)
    
    return X, y, fn, influential_indices, 'Sine Cosine'

def generate_dataset_poly_sine(n_samples=100, n_features=10, type="correlated"):
    influential_indices = np.arange(0, 2)
    X = generate_X(n_samples, n_features, influential_indices=influential_indices, correlation_value=0.6, type=type)

    def fn(X):
        f1, f2 = X[:, 0], X[:, 1]

        # Define the function using polynomial and sine terms
        y = f1**2 - 0.5 * f2**2 + np.sin(2 * np.pi * f1)
        
        return y

    y = fn(X)
    
    return X, y, fn, influential_indices, 'Poly Sine'

def generate_dataset_squared_exponentials(n_samples=100, n_features=10, type="correlated"):
    influential_indices = np.arange(0, 3)
    X = generate_X(n_samples, n_features, influential_indices=influential_indices, correlation_value=0.6, type=type)

    def fn(X):
        # Compute a function based on squared exponentials of the first 2 features
        y = np.exp(np.sum(X[:, :3]**2, axis=1) - 4.0)
        
        return y

    y = fn(X)
    
    return X, y, fn, influential_indices, 'Squared Exponentials'

# These functions are for more than 3 features

def generate_dataset_complex_tanhsin(n_samples=1000, n_features=10, type="correlated"):
    influential_indices = np.arange(0, 3)
    X = generate_X(n_samples, n_features, influential_indices=influential_indices, correlation_value=0.6, type=type)

    def fn(X):
        f1, f2, f3 = X[:, 0], X[:, 1], X[:, 2]

        # Main effects
        main_effect_1 = np.tanh(f1)  # Hyperbolic tangent effect
        main_effect_2 = np.abs(f2)  # Absolute value effect

        # Interaction effects
        interaction_effect_1 = f1 * f2  # Multiplicative interaction
        interaction_effect_2 = np.sin(f1 + f3)  # Nonlinear interaction

        # Combine effects
        y = main_effect_1 + main_effect_2 + interaction_effect_1 + interaction_effect_2
        return y

    y = fn(X)
    
    return X, y, fn, influential_indices, 'Tanh Sine'

def generate_dataset_complex_trig_exp(n_samples=100, n_features=10, type="correlated"):
    influential_indices = np.arange(0, 4)
    X = generate_X(n_samples, n_features, influential_indices=influential_indices, correlation_value=0.6, type=type)

    def fn(X):
        f1, f2, f3, f4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        # Complex non-linear interactions
        y = np.sin(f1) * np.exp(f2) + np.cos(f3 * f4) * np.tanh(f1 * f2 * np.pi)
        y += np.exp(-(f1**2 + f2**2)) * np.sin((f3 + f4)*np.pi)

        return y

    y = fn(X)

    return X, y, fn, influential_indices, 'Trigonometric Exponential'

def generate_dataset_complex_exponential_hyperbolic(n_samples=100, n_features=10, type="correlated"):
    influential_indices = np.arange(0, 4)
    X = generate_X(n_samples, n_features, influential_indices=influential_indices, correlation_value=0.6, type=type)

    def fn(X):
        f1, f2, f3, f4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        # Nested exponential and hyperbolic functions
        y = np.exp(f1) * np.tanh(f2 * f3) + np.exp(-np.abs(f4)) * np.tanh(f1 * f2)
        y += np.exp(f1 * f2) * np.sin(f3 * f4)

        return y

    y = fn(X)
    
    return X, y, fn, influential_indices, 'Exponential Hyperbolic'


def generate_dataset(data_name, n_samples=100, n_features=10, seed = 0, type="correlated"):
    np.random.seed(seed)
    if data_name == data_names[0]:
         X, y, fn, feature_imp, _ = generate_dataset_sinlog(n_samples, n_features, type)
    if data_name == data_names[1]:
         X, y, fn, feature_imp, _ = generate_dataset_sin(n_samples, n_features, type)
    if data_name == data_names[2]:
         X, y, fn, feature_imp, _ = generate_dataset_poly_sine(n_samples, n_features, type)
    if data_name == data_names[3]:
         X, y, fn, feature_imp, _ = generate_dataset_squared_exponentials(n_samples, n_features, type)
    if data_name == data_names[4]:
        X, y, fn, feature_imp, _ = generate_dataset_complex_tanhsin(n_samples, n_features, type)
    if data_name == data_names[5]:
        X, y, fn, feature_imp, _ = generate_dataset_complex_trig_exp(n_samples, n_features, type)
    if data_name == data_names[6]:
        X, y, fn, feature_imp, _ = generate_dataset_complex_exponential_hyperbolic(n_samples, n_features, type)
    
    Ground_Truth = Ground_Truth_Generation(X, data_name)
    return X, y, fn, feature_imp, Ground_Truth

def Ground_Truth_Generation(X, data_name):

    # Number of samples and features
    n = len(X[:,0])
    d = len(X[0,:])

    # Output initialization
    out = np.zeros([n,d])
   
    # Index
    if (data_name in data_names[0:3]):        
        out[:,:2] = 1
    
    elif(data_name in data_names[3:5]):        
        out[:,:3] = 1
    
    elif(data_name in data_names[5:7]):        
        out[:,:4] = 1
    elif(data_name in data_names[7]):        
        out[:,:5] = 1
    if (data_name in ['Syn4','Syn5','Syn6']):        
        idx1 = np.where(X[:,9]< 0)[0]
        idx2 = np.where(X[:,9]>=0)[0]
        out[:,9] = 1      
        out[idx1,:2] = 1
        out[idx2,2:6] = 1
          
    return out

def create_rank(scores): 
	"""
	Compute rank of each feature based on weight.
	
	"""
	scores = abs(scores)
	n, d = scores.shape
	ranks = []
	for i, score in enumerate(scores):
		# Random permutation to avoid bias due to equal weights.
		idx = np.random.permutation(d) 
		permutated_weights = score[idx]  
		permutated_rank=(-permutated_weights).argsort().argsort()+1
		rank = permutated_rank[np.argsort(idx)]

		ranks.append(rank)

	return np.array(ranks)
