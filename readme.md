# PKeX-Shapley: Computing Exact Shapley Value in Polynomial Time for Product-Kernel Methods

This repository accompanies the paper **"Computing Exact Shapley Value in Polynomial Time for Product-Kernel Methods"** and provides code for the PKeX-Shapley method. The demos illustrate how to use PKeX-Shapley for explaining product kernel learning methods, Maximum Mean Discrepancy (MMD), and Hilbert-Schmidt Independence Criterion (HSIC).

## Table of Contents

- [Product Kernel Learning Methods](#product-kernel-learning-methods)
- [Explaining MMD](#explaining-mmd)
- [Explaining HSIC](#explaining-hsic)
- [Bibliography](#bibliography)

---

## Product Kernel Learning Methods

**Demo:** `demo_localexplainer.ipynb`

This notebook demonstrates how to use PKeX-Shapley to explain predictions of models based on product kernels.

**Tutorial:**
1. **Load Data and Model:**
    ```python
    from pkexshapley import LocalExplainer
    # Load your data and trained kernel model
    ```
2. **Initialize the Explainer:**
    ```python
    explainer = LocalExplainer(model, X_train)
    ```
3. **Compute Shapley Values:**
    ```python
    shap_values = explainer.explain(X_test[0])
    print(shap_values)
    ```
4. **Visualize Results:**  
    The notebook provides code to plot and interpret the feature attributions.

---

## Explaining MMD

**Demo:** `demo_mmd.ipynb`

This notebook shows how to use PKeX-Shapley to attribute the Maximum Mean Discrepancy (MMD) between two distributions to individual features.

**Tutorial:**
1. **Prepare Two Datasets:**
    ```python
    from pkexshapley import MMDExplainer
    # X1, X2: datasets to compare
    ```
2. **Initialize the MMD Explainer:**
    ```python
    mmd_explainer = MMDExplainer(X1, X2)
    ```
3. **Compute Feature Attributions:**
    ```python
    mmd_shap = mmd_explainer.explain()
    print(mmd_shap)
    ```
4. **Interpretation:**  
    The output shows how much each feature contributes to the MMD.

---

## Explaining HSIC

**Demo:** `demo_hsic.ipynb`

This notebook demonstrates how to use PKeX-Shapley to explain the dependence between two variables as measured by HSIC.

**Tutorial:**
1. **Prepare Data:**
    ```python
    from pkexshapley import HSICExplainer
    # X, Y: variables to measure dependence
    ```
2. **Initialize the HSIC Explainer:**
    ```python
    hsic_explainer = HSICExplainer(X, Y)
    ```
3. **Compute Feature Attributions:**
    ```python
    hsic_shap = hsic_explainer.explain()
    print(hsic_shap)
    ```
4. **Interpretation:**  
    The results indicate the contribution of each feature to the HSIC value.

---

### Bibliography

```bibtex
@article{pkex_shapley,
    title={Computing Exact Shapley Values in Polynomial Time for Product-Kernel Methods},
    year={2025}
}
```

For further details, please refer to the individual demo notebooks.
