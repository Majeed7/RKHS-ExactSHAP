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
import os

from explainer.LocalExplainer import RBFLocalExplainer


results_xsl = Path('results/localexp_syn.xlsx')
if not os.path.exists(results_xsl):
    # Create an empty Excel file if it doesn't exist
    pd.DataFrame().to_excel(results_xsl, index=False)

# Define the number of samples and features as variables
mode='deploy'
n_samples = 1000
n_features = 20
n_trials = 100
tbx_no = 100

if mode == 'test':
    n_samples = 100
    n_features = 10
    n_trials = 10
    tbx_no = 2

# Generate synthesized datasets
datasets = [
    ("Squared Exponentials", generate_dataset_squared_exponentials(n_samples, n_features)),
    ("Polynomial Degree 5", generate_dataset_polynomial(n_samples, n_features, degree=5)),
    ("Polynomial Degree 10", generate_dataset_polynomial(n_samples, n_features, degree=10)),    
]
# Prepare a dictionary to store accuracies for each dataset
dataset_accuracies = {ds_name: {} for ds_name, _ in datasets}

# Prepare a dictionary to store execution times for each dataset
dataset_execution_times = {ds_name: {} for ds_name, _ in datasets}

# Iterate over datasets
for ds_name, (X, y, fn, feature_imp, ds) in datasets:
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    optimized_svm = optimize_svm_rbf(X, y, n_trials=n_trials)
    model = optimized_svm['model']
    fn = model.predict

    # Calculate training error for regression
    y_pred_train = model.predict(X)
    training_error = np.mean((y_pred_train - y) ** 2)  # Mean Squared Error
    print(f"Training Error (MSE) for {ds_name}: {training_error}")
    
    # Select 100 instances for explanation
    selected_indices = np.random.choice(X.shape[0], size=tbx_no, replace=False)
    X_tbx = X[selected_indices, :]

    # Initialize results storage for this dataset
    dataset_accuracies[ds_name] = {}
    dataset_execution_times[ds_name] = {}

    # RBFLocalExplainer
    explainer_recursive = RBFLocalExplainer(model)
    start_time = datetime.now()
    shapley_values_recursive = [explainer_recursive.explain(x) for x in X_tbx]
    end_time = datetime.now()
    dataset_execution_times[ds_name]["RBFLocalExplainer"] = (end_time - start_time).total_seconds()
    dataset_accuracies[ds_name]["RBFLocalExplainer"] = [
        len(set(feature_imp) & set(np.argsort(-np.abs(shapley_values_recursive[i]))[:len(feature_imp)])) / len(feature_imp)
        for i in range(len(X_tbx))
    ]

    # Compare methods
    for nsamples in [500, 1000]:
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
with pd.ExcelWriter(results_xsl, engine="openpyxl", mode="w") as writer:
    for ds_name in dataset_accuracies:
        # Create a DataFrame for accuracies
        accuracy_df = pd.DataFrame(dataset_accuracies[ds_name])
        accuracy_df.to_excel(writer, sheet_name=f"{ds_name}_Accuracies", index=False)

        # Create a DataFrame for execution times
        execution_time_df = pd.DataFrame(list(dataset_execution_times[ds_name].items()), columns=["Method", "Execution Time"])
        execution_time_df.to_excel(writer, sheet_name=f"{ds_name}_ExecutionTimes", index=False)
