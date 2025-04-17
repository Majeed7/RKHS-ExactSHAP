import numpy as np
from scipy.stats import wishart, t
from explainer.MMDExplainer import MMDExplainer
import matplotlib.pyplot as plt
import seaborn as sns

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


# Set parameters
d = 20  # Total number of variables
d_prime = 10  # Number of variables for the first distribution
n_samples = 1000  # Number of instances

mode = 'indepedent'

# Generate the covariance matrix for the first d' features
wishart_df1 = d_prime  # Degrees of freedom for Wishart distribution
wishart_scale1 = np.eye(d_prime)  # Scale matrix for Wishart distribution
cov1 = wishart.rvs(df=wishart_df1, scale=wishart_scale1, size=1)

if mode == 'indepedent':
    cov1 = np.eye(d_prime)  # Independent features

# Generate the first d' features based on multivariate normal distribution
mean1 = np.zeros(d_prime)  # Mean vector
X1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=n_samples)

# Generate the remaining d-d' features based on multivariate normal distribution
mean2 = np.zeros(d - d_prime)  # Mean vector
X2 = np.random.multivariate_normal(mean=mean2, cov=np.eye(d-d_prime), size=n_samples)

# Combine the two parts to form the full dataset X
X = np.hstack((X1, X2))

# Generate the second dataset Y
# The first d' features are the same as in X
Z1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=n_samples)

# The remaining d-d' features are sampled from a Student's t-distribution
# Match the moments of the features in X2
Z2_mean = np.mean(X2, axis=0)
Z2_std = np.std(X2, axis=0)
Z2 = t.rvs(df=3, loc=Z2_mean, scale=Z2_std, size=(n_samples, d - d_prime))

# Combine the two parts to form the full dataset Y
Z = np.hstack((Z1, Z2))

explainer = MMDExplainer(X=X, Z=Z)
sv = explainer.explain()

plt.figure(figsize=(10, 5)) 
variables = [f"V{i}" for i in range(1,d+1)]
sv_statistics = n_samples * np.array(sv)
ax = sns.barplot(x=variables, y=sv_statistics)

ax.set_title(f"Shapley Value Statistics for Variables")
ax.set_xlabel("Variables")
ax.set_ylabel("Shapley Values")

plt.tight_layout()
plt.show()
plt.savefig('results/hypothesis_testing.png', dpi=500, format='png', bbox_inches='tight')

## Equal distribution and Shapley value
explainer2 = MMDExplainer(X=X1, Z=Z1)
sv2 = explainer2.explain()

plt.figure(figsize=(10, 5)) 
variables2 = [f"V{i}" for i in range(1,d_prime+1)]
sv_statistics2 = n_samples * np.array(sv2)
ax = sns.barplot(x=variables2, y=sv_statistics2)

ax.set_title(f"Shapley Value Statistics for Wine Dataset")
ax.set_xlabel("Features")
ax.set_ylabel("Shapley Values")


plt.tight_layout()
plt.show()
plt.savefig('results/hypothesis_testing_equaldist.png', dpi=500, format='png', bbox_inches='tight')


'''
 Apply on Wine Dataset
'''
from sklearn.datasets import load_diabetes

data = load_diabetes()
X = data['data']
y = data['target']
feature_names = data['feature_names']

# Split X based on the second feature (sex)
X_male = X[X[:, 1] > 0]  # Assuming 1 represents male
X_female = X[X[:, 1] < 0]  # Assuming 0 represents female

X_male = np.delete(X_male, 1, axis=1)
X_female = np.delete(X_female, 1, axis=1)
sel_fs = np.delete(feature_names, 1)

explainer_wine = MMDExplainer(X=X_male, Z=X_female)
sv_wine = explainer_wine.explain()
sv_wine_statistics = X_male.shape[0] * np.array(sv_wine)

plt.figure(figsize=(10, 5)) 
ax = sns.barplot(x=sel_fs, y=sv_wine_statistics)

ax.set_title(f"Shapley Value Statistics for Variables")
ax.set_xlabel("Variables")
ax.set_ylabel("Shapley Values")


plt.tight_layout()
plt.show()
plt.savefig('results/hypothesis_testing_wine.png', dpi=500, format='png', bbox_inches='tight')

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

# Loop over each column/feature
for i, ax in enumerate(axes.flatten()):
    sns.distplot(X_male[:, i], label="male", ax=ax, kde=True, hist=True)
    sns.distplot(X_female[:, i], label="female", ax=ax, kde=True, hist=True)
    ax.set_title(f"Feature {i}")
    ax.legend()
    ax.set_title(sel_fs[i])

plt.tight_layout()
plt.show()

plt.savefig('results/hypothesis_testing_wine_distributions.png', dpi=500, format='png', bbox_inches='tight')
