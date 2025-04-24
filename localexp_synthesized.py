import numpy as np
import matplotlib.pyplot as plt 

import shap 
from otherexplainers.gemfix import GEMFIX
from otherexplainers.bishapley_kernel import Bivariate_KernelExplainer
from otherexplainers.MAPLE import MAPLE
from lime import lime_tabular

from pathlib import Path
import pandas as pd 
from openpyxl import load_workbook
from sklearn.preprocessing import StandardScaler

#from synthesized_datasets import *
from utils.synthesized_datasets import *
from utils.svm_training import *
from datetime import datetime
import time
import os

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.linalg import solve_triangular

from explainer.LocalExplainer import RBFLocalExplainer, ProductKernelLocalExplainer


results_xsl = Path(f'results/localexp_syn_{time.strftime("%Y%m%d_%H%M%S")}.xlsx')


'''
Training A Gaussian Process Regressor
'''
def train_gp(X, y):
    """
    Train a Gaussian Process model for the given input X and y.
    The RBF kernel is adapted to the number of features in X.
    """
    # Define kernel with individual length scales for each feature and noise term
    n_features = X.shape[1]
    kernel =  RBF(length_scale=np.ones(n_features)) + WhiteKernel(noise_level=0.1)
    
    # Initialize and train GP
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
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
        feature_kernels.append(kernel.squeeze())
    
    return np.array(feature_kernels)




# Define the number of samples and features as variables
mode='deploy'
n_samples = 1000
n_features = 25
n_trials = 100
tbx_no = 10

if mode == 'test':
    n_samples = 100
    n_features = 10
    n_trials = 10
    tbx_no = 2

# Generate synthesized datasets
datasets = [
    ("SinLog", generate_dataset_sinlog(n_samples, n_features)),
     ("Squared Exponentials", generate_dataset_squared_exponentials(n_samples, n_features)),
    ("Polynomial Degree 10", generate_dataset_polynomial(n_samples, n_features, degree=10)),    
    ("Polynomial Degree 5", generate_dataset_polynomial(n_samples, n_features, degree=5)),
]
# Prepare a dictionary to store accuracies for each dataset
dataset_accuracies = {ds_name: {} for ds_name, _ in datasets}

# Prepare a dictionary to store execution times for each dataset
dataset_execution_times = {ds_name: {} for ds_name, _ in datasets}

# Iterate over datasets
for ds_name, (X, y, fn, feature_imp, ds) in datasets:
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    gp = train_gp(X, y)
    constant_value = 1 #gp.kernel_.k1.k1.constant_value
    rbf_length_scales = gp.kernel_.k1.length_scale #.k2.length_scale
    noise_level = gp.kernel_.k2.noise_level


    # Select 100 instances for explanation
    selected_indices = np.random.choice(X.shape[0], size=tbx_no, replace=False)
    X_tbx = X[selected_indices, :]

    # Initialize results storage for this dataset
    dataset_accuracies[ds_name] = {}
    dataset_execution_times[ds_name] = {}

    # RBFLocalExplainer
    # explainer_recursive = RBFLocalExplainer(model)
    #start_time = datetime.now()
    #shapley_values_recursive = [explainer_recursive.explain(x) for x in X_tbx]
    #end_time = datetime.now()

    explainer_recursive = ProductKernelLocalExplainer(gp)
    shapley_values_recursive = []
    start_time = datetime.now()
    for x in X_tbx:
        kernel_vec = compute_feature_wise_rbf_kernel(x.reshape(1,-1), X_train=X, length_scales=rbf_length_scales, constant_value=constant_value)
        sv = explainer_recursive.explain_by_kernel_vectors(kernel_vectors=kernel_vec)
        shapley_values_recursive.append(sv)
    dataset_execution_times[ds_name]["RBFLocalExplainer"] = (datetime.now() - start_time).total_seconds()
    dataset_accuracies[ds_name]["RBFLocalExplainer"] = [
        len(set(feature_imp) & set(np.argsort(-np.abs(shapley_values_recursive[i]))[:len(feature_imp)])) / len(feature_imp)
        for i in range(len(X_tbx))
    ]

    # Compare methods
    for nsamples in [200, 1000]:
        # GEMFIX
        X_bg = shap.sample(X, 100)
        gemfix = GEMFIX(fn, X_bg, lam=0.001)
        start_time = datetime.now()
        gem_values = gemfix.shap_values(X_tbx, nsamples=nsamples)
        end_time = datetime.now()
        dataset_execution_times[ds_name][f"GEMFIX({nsamples})"] = (end_time - start_time).total_seconds()
        dataset_accuracies[ds_name][f"GEMFIX({nsamples})"] = [
            len(set(feature_imp) & set(np.argsort(-np.abs(gem_values[i]))[:len(feature_imp)])) / len(feature_imp)
            for i in range(len(X_tbx))
        ]

        # SHAP
        explainer = shap.KernelExplainer(fn, X_bg, l1_reg=False)
        start_time = datetime.now()
        shap_values = explainer.shap_values(X_tbx, nsamples=nsamples, l1_reg=False)
        end_time = datetime.now()
        dataset_execution_times[ds_name][f"Kernel SHAP({nsamples})"] = (end_time - start_time).total_seconds()
        dataset_accuracies[ds_name][f"Kernel SHAP({nsamples})"] = [
            len(set(feature_imp) & set(np.argsort(-np.abs(shap_values[i]))[:len(feature_imp)])) / len(feature_imp)
            for i in range(len(X_tbx))
        ]

        # BiShap
        bivariate_explainer = Bivariate_KernelExplainer(fn, X_bg)
        start_time = datetime.now()
        bishap_values = bivariate_explainer.shap_values(X_tbx, nsamples=nsamples)
        end_time = datetime.now()
        dataset_execution_times[ds_name][f"BiShap({nsamples})"] = (end_time - start_time).total_seconds()
        dataset_accuracies[ds_name][f"BiShap({nsamples})"] = [
            len(set(feature_imp) & set(np.argsort(-np.abs(bishap_values[i]))[:len(feature_imp)])) / len(feature_imp)
            for i in range(len(X_tbx))
        ]

        # Sampling SHAP
        sexplainer = shap.SamplingExplainer(fn, X_bg, l1_reg=False)
        start_time = datetime.now()
        sshap_values = sexplainer.shap_values(X_tbx, nsamples=nsamples, l1_reg=False)
        end_time = datetime.now()
        dataset_execution_times[ds_name][f"Sampling SHAP({nsamples})"] = (end_time - start_time).total_seconds()
        dataset_accuracies[ds_name][f"Sampling SHAP({nsamples})"] = [
            len(set(feature_imp) & set(np.argsort(-np.abs(sshap_values[i]))[:len(feature_imp)])) / len(feature_imp)
            for i in range(len(X_tbx))
        ]

# Save results to Excel
if not os.path.exists(results_xsl):
    # Create an empty Excel file if it doesn't exist
    pd.DataFrame().to_excel(results_xsl, index=False)

with pd.ExcelWriter(results_xsl, engine="openpyxl", mode="w") as writer:
    for ds_name in dataset_accuracies:
        # Create a DataFrame for accuracies
        accuracy_df = pd.DataFrame(dataset_accuracies[ds_name])
        accuracy_df.to_excel(writer, sheet_name=f"{ds_name}_Accuracies", index=False)

        # Create a DataFrame for execution times
        execution_time_df = pd.DataFrame(list(dataset_execution_times[ds_name].items()), columns=["Method", "Execution Time"])
        execution_time_df.to_excel(writer, sheet_name=f"{ds_name}_ExecutionTimes", index=False)
