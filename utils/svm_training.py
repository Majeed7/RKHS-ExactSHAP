import optuna
import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import type_of_target
from typing import Dict, Any


def determine_problem_type(y):
    target_type = type_of_target(y)
    if target_type in ['continuous', 'continuous-multioutput']:
        return 'regression'
    else:
        return 'classification'


def objective_regression(trial, X, y):
    # Define hyperparameters to optimize for RBF kernel
    C = trial.suggest_loguniform('C', 1e-5, 1e5)  # Expanded range for C
    gamma = trial.suggest_loguniform('gamma', 1e-5, 1e3)  # Expanded range for gamma

    # Use cross-validation only for hyperparameter tuning
    model = SVR(kernel='rbf', C=C, gamma=gamma)
    score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    return -np.mean(score)


def objective_classification(trial, X, y):
    # Define hyperparameters to optimize for RBF kernel
    C = trial.suggest_loguniform('C', 1e-5, 1e5)  # Expanded range for C
    gamma = trial.suggest_loguniform('gamma', 1e-5, 1e3)  # Expanded range for gamma

    # Use cross-validation only for hyperparameter tuning
    model = SVC(kernel='rbf', C=C, gamma=gamma)
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return np.mean(score)


def optimize_svm_rbf(X: np.ndarray,
                     y: np.ndarray,
                     n_trials: int = 100) -> Dict[str, Any]:
    """
    Optimize SVM with RBF kernel using Optuna for hyperparameter tuning.

    Parameters:
    ----
    X : np.ndarray
         Feature matrix
    y : np.ndarray
        Target variable
    n_trials : int
        Number of optimization trials

    Returns:
    ----
    Dict[str, Any] with:
        - 'model': Fitted SVM model with best parameters
        - 'best_params': Best hyperparameters found
        - 'best_score': Best CV score during optimization
        - 'problem_type': 'regression' or 'classification'
    """

    # Determine problem type
    problem_type = determine_problem_type(y)
    print(f"Detected problem type: {problem_type}")

    # Create study object
    if problem_type == 'regression':
        study = optuna.create_study(direction='minimize')  # Minimize MSE
        objective = lambda trial: objective_regression(trial, X, y)
    else:
        study = optuna.create_study(direction='maximize')  # Maximize accuracy
        objective = lambda trial: objective_classification(trial, X, y)

    # Optimize hyperparameters
    study.optimize(objective, n_trials=n_trials)

    # Get best parameters and fit final model
    best_params = study.best_params
    if problem_type == 'regression':
        final_model = SVR(kernel='rbf', **best_params)
    else:
        final_model = SVC(kernel='rbf', **best_params)

    # Fit the model on the entire dataset
    final_model.fit(X, y)

    return {
        'model': final_model,  # Fitted model ready for predictions
        'best_params': best_params,  # Best hyperparameters found
        'best_score': study.best_value,  # Best CV score during optimization
        'problem_type': problem_type  # Type of problem detected
    }

# Example usage:
"""
# Assuming X is already scaled and y is your target
results = optimize_svm_rbf(X, y, n_trials=100)

# Access results
model = results['model']  # Final fitted model
best_params = results['best_params']  # Best hyperparameters
best_score = results['best_score']  # Best CV score during optimization

# Make predictions
predictions = model.predict(X_new)

# Print results
print(f"Best parameters: {best_params}")
print(f"Best CV score during optimization: {best_score}")
"""