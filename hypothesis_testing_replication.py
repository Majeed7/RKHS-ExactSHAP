import numpy as np
from scipy.stats import wishart, t
from explainer.MMDExplainer import MMDExplainer
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

## plot settings    
plt.rcParams.update({
    'font.size': 10,     # Set default font size
    'font.weight': 'bold',  # Set default font weight to bold
    'axes.labelweight': 'bold',  # Ensure the axis labels are bold
    'axes.titleweight': 'bold',  # Ensure the titles are bold
    'figure.titleweight': 'bold',  # Bold for suptitle if you use fig.suptitle()
    'xtick.labelsize': 8,  # Font size for X-tick labels
    'ytick.labelsize': 12,  # Font size for Y-tick labels
    'xtick.major.size': 5,  # Length of major ticks
    'ytick.major.size': 5,  # Length of major ticks
    'xtick.minor.size': 3,  # Length of minor ticks
    'ytick.minor.size': 3   # Length of minor ticks
})

n_bins = 20

# Set parameters
d = 20  # Total number of variables
d_prime = 10  # Number of variables for the first distribution
n_samples = 1000  # Number of instances
n_trials = 1000  # Number of trials

estimation_type = "V"
mode = 'independent'

plot_only = False

if not plot_only:

    if len(sys.argv) < 2:
        print("No argument passed. Using default value: 1")
        mode = "independent"
    else:
        # Retrieve the parameter passed from Bash
        parameter = sys.argv[1]

        # Try converting the argument to a number
        try:
            # Try converting to an integer
            mode_index = int(parameter)
            mode = "independent" if mode_index == 0 else "dependent"

        except ValueError:
            # If it fails, try converting to a float
            mode = "independent"
            print("Cannot process the value. Using default value: 0.1")



    # Initialize storage for Shapley values
    shapley_values_case1 = []
    shapley_values_case2 = []

    for trial in range(n_trials):
        # Generate the covariance matrix for the first d' features
        wishart_df1 = d_prime  # Degrees of freedom for Wishart distribution
        wishart_scale1 = np.eye(d_prime)  # Scale matrix for Wishart distribution
        cov1 = wishart.rvs(df=wishart_df1, scale=wishart_scale1, size=1)

        if mode == 'independent':
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
        explainer = MMDExplainer(X=X, Z=Y, estimation_type=estimation_type)
        sv_case1 = explainer.explain()
        sv_case1 = n_samples * np.array(sv_case1)
        shapley_values_case1.append(sv_case1)

        # Case 2: Compute Shapley values for X1 and Y1
        explainer2 = MMDExplainer(X=X1, Z=Y1, estimation_type=estimation_type)
        sv_case2 = explainer2.explain()
        sv_case2 = n_samples * np.array(sv_case2)
        shapley_values_case2.append(sv_case2)

    # Convert results to DataFrames
    df_case1 = pd.DataFrame(shapley_values_case1, columns=[f"V{i}" for i in range(1, d + 1)])
    df_case2 = pd.DataFrame(shapley_values_case2, columns=[f"V{i}" for i in range(1, d_prime + 1)])

    # Save results to an Excel file with two sheets
    output_file = f"results/mmd/hypothesis_testing_sv_{mode}_{estimation_type}stat.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        df_case1.to_excel(writer, sheet_name="Case1_X_vs_Y", index=False)
        df_case2.to_excel(writer, sheet_name="Case2_X1_vs_Y1", index=False)

    print(f"Shapley values saved to {output_file}")

    import matplotlib.pyplot as plt

    # Create the results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

# Load the data from the Excel file
loaded_file = f"results/mmd/hypothesis_testing_sv_{mode}_{estimation_type}stat.xlsx"
df_case1_loaded = pd.read_excel(loaded_file, sheet_name="Case1_X_vs_Y")
df_case2_loaded = pd.read_excel(loaded_file, sheet_name="Case2_X1_vs_Y1")

# Plot histograms for Case 1 from the loaded data
n_cols = min(5, d)
n_rows = (d + n_cols - 1) // n_cols
fig_case1_loaded, axes_case1_loaded = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
axes_case1_loaded = axes_case1_loaded.flatten()

# Determine the global x-axis range
x_min = min(df_case1_loaded.min())
x_max = max(df_case1_loaded.max())

for i in range(d):
    axes_case1_loaded[i].hist(df_case1_loaded[f"V{i+1}"], bins=n_bins, alpha=0.7, edgecolor='black')
    axes_case1_loaded[i].set_title(f"Variable {i+1}")
    #axes_case1_loaded[i].set_xlabel("Shapley Value")
    #axes_case1_loaded[i].grid(True)
    #axes_case1_loaded[i].set_xlim(x_min, x_max)  # Set the x-axis range

for i in range(d):
    if i % 5 == 0:
        axes_case1_loaded[i].set_ylabel("Frequency")

for i in range(d, len(axes_case1_loaded)):
    axes_case1_loaded[i].axis('off')

fig_case1_loaded.savefig(f"results/mmd/hypothesis_testing_histograms_case1_{mode}_{estimation_type}stat.png", dpi=500, format='png', bbox_inches='tight')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
plt.close()

# Plot histograms for Case 2 from the loaded data
n_cols = min(5, d_prime)
n_rows = (d_prime + n_cols - 1) // n_cols
fig_case2_loaded, axes_case2_loaded = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
axes_case2_loaded = axes_case2_loaded.flatten()

# Determine the global x-axis range for consistency
x_min = min(df_case2_loaded.min())
x_max = max(df_case2_loaded.max())

for i in range(d_prime):
    axes_case2_loaded[i].hist(df_case2_loaded[f"V{i+1}"], bins=n_bins, alpha=0.7, edgecolor='black')
    axes_case2_loaded[i].set_title(f"Variable {i+1}")
    axes_case2_loaded[i].tick_params(axis='x', labelsize=6.5)  # Adjust x-axis tick label size
    axes_case2_loaded[i].tick_params(axis='y', labelsize=8)  # Adjust y-axis tick label size
    axes_case2_loaded[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))  # Format x-axis ticks
    #axes_case2_loaded[i].grid(True)
    axes_case2_loaded[i].set_xlim(x_min, x_max)  # Set the x-axis range

for i in range(d_prime):
    if i % 5 == 0:
        axes_case2_loaded[i].set_ylabel("Frequency", fontsize=8)  # Set smaller font size for y-axis labels

for i in range(d_prime, len(axes_case2_loaded)):
    axes_case2_loaded[i].axis('off')

# Adjust layout to prevent label overlap
plt.tight_layout()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
fig_case2_loaded.savefig(f"results/mmd/hypothesis_testing_histograms_case2_{mode}_{estimation_type}stat.png", dpi=500, format='png', bbox_inches='tight')
plt.close()

print("Histograms of loaded Shapley values saved in the 'results' folder.")


