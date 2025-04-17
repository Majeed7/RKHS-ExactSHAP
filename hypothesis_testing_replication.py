import numpy as np
from scipy.stats import wishart, t
from explainer.MMDExplainer import MMDExplainer
import pandas as pd
import os
import sys

if len(sys.argv) < 2:
    print("No argument passed. Using default value: 1")
    mode = "indepedent"
else:
    # Retrieve the parameter passed from Bash
    parameter = sys.argv[1]

    # Try converting the argument to a number
    try:
        # Try converting to an integer
        mode_index = int(parameter)
        mode = "indepedent" if mode_index == 0 else "dependent"

    except ValueError:
        # If it fails, try converting to a float
        mode = "indepedent"
        print("Cannot process the value. Using default value: 0.1")


# Set parameters
d = 20  # Total number of variables
d_prime = 10  # Number of variables for the first distribution
n_samples = 1000  # Number of instances
n_trials = 50  # Number of trials

# Initialize storage for Shapley values
shapley_values_case1 = []
shapley_values_case2 = []

for trial in range(n_trials):
    # Generate the covariance matrix for the first d' features
    wishart_df1 = d_prime  # Degrees of freedom for Wishart distribution
    wishart_scale1 = np.eye(d_prime)  # Scale matrix for Wishart distribution
    cov1 = wishart.rvs(df=wishart_df1, scale=wishart_scale1, size=1)

    if mode == 'indepedent':
        cov1 = np.eye(d_prime)  # Independent features

    # Generate the first d' features based on multivariate normal distribution
    mean1 = np.zeros(d_prime)  # Mean vector
    X1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=n_samples)

    # Generate the covariance matrix for the remaining d-d' features
    wishart_df2 = d - d_prime  # Degrees of freedom for Wishart distribution
    wishart_scale2 = np.eye(d - d_prime)  # Scale matrix for Wishart distribution
    cov2 = wishart.rvs(df=wishart_df2, scale=wishart_scale2, size=1)

    # Generate the remaining d-d' features based on multivariate normal distribution
    mean2 = np.zeros(d - d_prime)  # Mean vector
    X2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=n_samples)

    # Combine the two parts to form the full dataset X
    X = np.hstack((X1, X2))

    # Generate the second dataset Y
    # The first d' features are the same as in X
    Y1 = X1

    # The remaining d-d' features are sampled from a Student's t-distribution
    # Match the moments of the features in X2
    X2_mean = np.mean(X2, axis=0)
    X2_std = np.std(X2, axis=0)
    Y2 = t.rvs(df=3, loc=X2_mean, scale=X2_std, size=(n_samples, d - d_prime))

    # Combine the two parts to form the full dataset Y
    Y = np.hstack((Y1, Y2))

    # Case 1: Compute Shapley values for X and Y
    explainer = MMDExplainer(X=X, Z=Y)
    sv_case1 = explainer.explain()
    sv_case1 = n_samples * np.array(sv_case1)
    shapley_values_case1.append(sv_case1)

    # Case 2: Compute Shapley values for X1 and Y1
    explainer2 = MMDExplainer(X=X1, Z=Y1)
    sv_case2 = explainer2.explain()
    sv_case2 = n_samples * np.array(sv_case2)
    shapley_values_case2.append(sv_case2)

# Convert results to DataFrames
df_case1 = pd.DataFrame(shapley_values_case1, columns=[f"V{i}" for i in range(1, d + 1)])
df_case2 = pd.DataFrame(shapley_values_case2, columns=[f"V{i}" for i in range(1, d_prime + 1)])

# Save results to an Excel file with two sheets
output_file = f"results/hypothesis_testing_sv_{mode}.xlsx"
with pd.ExcelWriter(output_file) as writer:
    df_case1.to_excel(writer, sheet_name="Case1_X_vs_Y", index=False)
    df_case2.to_excel(writer, sheet_name="Case2_X1_vs_Y1", index=False)

print(f"Shapley values saved to {output_file}")

import matplotlib.pyplot as plt

# Create the results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Plot histograms for Case 1 in a single figure with subplots
# Plot histograms for Case 1 in a figure with multiple rows and at most 5 plots per row
n_cols = min(5, d)
n_rows = (d + n_cols - 1) // n_cols  # Calculate the number of rows needed
fig_case1, axes_case1 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
axes_case1 = axes_case1.flatten()  # Flatten in case of multiple rows
for i in range(d):
    axes_case1[i].hist(df_case1[f"V{i+1}"], bins=10, alpha=0.7, color='blue', edgecolor='black')
    axes_case1[i].set_title(f"Feature V{i+1}")
    axes_case1[i].set_xlabel("Shapley Value")
    axes_case1[i].grid(True)
axes_case1[0].set_ylabel("Frequency")
for i in range(d, len(axes_case1)):  # Hide unused subplots
    axes_case1[i].axis('off')
fig_case1.suptitle("Histograms of Shapley Values")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"results/hypothesis_testing_histograms_case1_{mode}.png")
plt.close()

# Plot histograms for Case 2 in a figure with multiple rows and at most 5 plots per row
n_cols = min(5, d_prime)
n_rows = (d_prime + n_cols - 1) // n_cols  # Calculate the number of rows needed
fig_case2, axes_case2 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
axes_case2 = axes_case2.flatten()  # Flatten in case of multiple rows
for i in range(d_prime):
    axes_case2[i].hist(df_case2[f"V{i+1}"], bins=10, alpha=0.7, color='green', edgecolor='black')
    axes_case2[i].set_title(f"Feature V{i+1}")
    axes_case2[i].set_xlabel("Shapley Value")
    axes_case2[i].grid(True)
axes_case2[0].set_ylabel("Frequency")
for i in range(d_prime, len(axes_case2)):  # Hide unused subplots
    axes_case2[i].axis('off')
fig_case2.suptitle("Histograms of Shapley Values")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"results/hypothesis_testing_histograms_case2_{mode}.png")
plt.show()
plt.close()

print("Combined histograms of Shapley values saved in the 'results' folder.")