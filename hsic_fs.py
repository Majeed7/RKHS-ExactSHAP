from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils.multiclass import type_of_target
import copy 
import shap 

import os
import pickle
import time
from openpyxl import Workbook
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest, f_classif, f_regression, RFECV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from pathlib import Path
import gc
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy import io 
from pyHSICLasso import HSICLasso

from scipy.cluster.vq import kmeans
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from ucimlrepo import fetch_ucirepo 

from utils.real_datasets import load_dataset
from explainer.HSICExplainer import HSICExplainer
import dill

import warnings
warnings.filterwarnings("ignore")

def train_svm(X_train, y_train, X_test, y_test):
    """
    Train an SVM or SVR model with imputation for missing values.

    Parameters:
        X_train: Training feature matrix
        y_train: Training labels
        X_test: Testing feature matrix
        y_test: Testing labels

    Returns:
        best_model: Trained model after hyperparameter tuning
        best_params: Best hyperparameters from GridSearchCV
        score: Performance score (accuracy for classification, RMSE for regression)
    """

    # Check the type of the target variable
    target_type = type_of_target(y_train)
    is_classification = target_type in ["binary", "multiclass"]

    # Define the parameter grid
    param_grid = {
        "svm__C": [0.1, 1, 10],  # Regularization parameter
    }

    # Choose the model
    model = SVC() if is_classification else SVR()

    # Create a pipeline with an imputer and the SVM/SVR model
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),  # Impute missing values
        ("svm", model)
    ])

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy" if is_classification else "neg_mean_squared_error")
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    if is_classification:
        score = accuracy_score(y_test, y_pred)
    else:
        rmse = mean_squared_error(y_test, y_pred)
        score = rmse

    print("Best Parameters:", best_params)
    print("Performance Score:", score)

    return best_model, best_params, score

# Ensure a directory exists for saving models
os.makedirs("trained_models", exist_ok=True)

# Define the list of feature selectors
feature_selectors = ["HSIC-SV"] # [", "Sobol",HSICLasso", "mutual_info", "lasso", "k_best", "tree_ensemble"] #["AGP-SHAP", "Sobol",] #, "rfecv"]

# Initialize an Excel workbook to store global importance values
wb = Workbook()




if __name__ == '__main__':
    # steel: 1941 * 33    binary
    # waveform: 5000 * 40 binary 
    # sonar: 208 * 61 binary 
    
    # nomao: 34465 * 118 binary
    #did not work on these datasets: #"steel", "ionosphere", "gas", "pol", "sml"]
    
    dataset_names = ["breast_cancer", "sonar", "waveform"] #"nomao" did not work for HSIC
    dataset_names2 = ["breast_cancer_wisconsin", "skillcraft"]
    dataset_names3 = ['parkinson', 'keggdirected', "pumadyn32nm", "crime"]    # Main running part of the script
    for dataset_name in dataset_names2:
        print(f"\nProcessing dataset: {dataset_name}")
        try:
            X, y = load_dataset(dataset_name)
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            continue

        # Determine if the dataset is for classification or regression
        mode = "classification" if type_of_target(y) in ["binary", "multiclass"] else "regression"

        if mode != "regression":
            label_encoder = LabelEncoder()
            label_encoder.fit_transform(y)
            y = label_encoder.fit_transform(y).reshape(-1, 1)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        n, d = X_train.shape
        
        '''
        Train Support Vector Machine with RBF kernel
        '''
        # Train SVM on the full dataset and store the best model
        print("Training SVM on the full dataset...")
        best_model, best_params, full_score = train_svm(X_train, y_train, X_test, y_test)

        # Save the trained model to a file
        model_filename = f"trained_models/svm_{dataset_name}.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(best_model, f)
        print(f"Saved best SVM model for {dataset_name} to {model_filename}")

        # Prepare an Excel sheet for the current dataset
        sheet = wb.create_sheet(title=dataset_name)
        sheet.append(["Feature Selector", "Execution Time"] + [f"Feature {i}" for i in range(X.shape[1])])

        # Apply each feature selector
        for selector in feature_selectors:
            print(f"Applying feature selector: {selector} on dataset: {dataset_name}")
            start_time = time.time()

            if selector == "HSIC-SV":
                hsicx = HSICExplainer(X_train, y_train)
                hsic_sv = hsicx.explain()

            elif selector == "HSICLasso":
                hsic_lasso = HSICLasso()
                hsic_lasso.input(X_train,y_train.squeeze())
                if mode == "classification": hsic_lasso.classification(d, covars=X_train) 
                else: hsic_lasso.regression(d, covars=X_train)
                hsic_ind = hsic_lasso.get_index()
                init_ranks = (len(hsic_ind) + (d - 1/2 - len(hsic_ind))/2) * np.ones((d,))
                init_ranks[hsic_ind] = np.arange(1,len(hsic_ind)+1)
                global_importance = d - init_ranks 

            elif selector == "mutual_info":
                global_importance = mutual_info_classif(X_train, y_train) if mode == "classification" else mutual_info_regression(X_train, y_train)

            elif selector == "lasso":
                lasso = LassoCV(alphas=None, cv=5, random_state=42).fit(X_train, y_train)
                global_importance = np.abs(lasso.coef_)

            elif selector == "rfecv":
                estimator = SVC(kernel="linear") if mode == "classification" else SVR(kernel="linear")
                rfecv = RFECV(estimator, step=1, cv=5)
                rfecv.fit(X_train, y_train)
                global_importance = rfecv.ranking_

            elif selector == "k_best":
                bestfeatures = SelectKBest(score_func=f_classif, k="all") if mode == "classification" else SelectKBest(score_func=f_regression, k="all")
                fit = bestfeatures.fit(X_train, y_train)
                global_importance = fit.scores_

            elif selector == "tree_ensemble":
                model = ExtraTreesClassifier(n_estimators=50) if mode == "classification" else ExtraTreesRegressor(n_estimators=50)
                model.fit(X_train, y_train)
                global_importance = model.feature_importances_

            else:
                print(f"Unknown feature selector: {selector}")
                continue

            execution_time = time.time() - start_time
            print(f"Execution time for {selector}: {execution_time}")

            # Store global importance values in the Excel sheet
            sheet.append([selector, execution_time] + list(global_importance))

        # Save the Excel file after processing each dataset
        excel_filename = "feature_importance_1.4.xlsx"
        wb.save(excel_filename)
        print(f"Global feature importance for {dataset_name} saved to {excel_filename}")
    wb.close()
    print("All datasets processed!")
    