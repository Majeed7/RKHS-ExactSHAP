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


results_xsl = Path('local_explanation_synthesized.xlsx')
if not os.path.exists(results_xsl):
    # Create an empty Excel file if it doesn't exist
    pd.DataFrame().to_excel(results_xsl, index=False)


if __name__ == '__main__':
    np.random.seed(30)

    X_sample_no = 200  # number of sampels for generating explanation
    sample_tbX = 10   # number of samples to be explained
    sample_no_gn = 1000 # number of generated synthesized instances 
    feature_no_gn = 10 # number of features for the synthesized instances

    # Example usage of one of the functions
    datasets=['Sine Log', 'Sine Cosine', 'Poly Sine', 'Squared Exponentials', 'Tanh Sine', \
              'Trigonometric Exponential', 'Exponential Hyperbolic', 'XOR']
    # Sine Log, Poly Sine, Squared Exponentials, 
    for ds_name in datasets:
        #X, y, fn, feature_imp, ds_name = generate_data(n=sample_no_gn, d=feature_no_gn, datatype=ds)
        X, y, fn, feature_imp, g_train = generate_dataset(ds_name, sample_no_gn, feature_no_gn, 42, type="independent")
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        selected_indices = np.random.choice(X.shape[0], size=sample_tbX, replace=False)
        X_tbx = X[selected_indices, :]

        n,d = X.shape

        optimized_svm = optimize_svm_rbf(X, y, n_trials=50)

        model = optimized_svm['model']
        fn = model.predict

        from explainer.LocalExplainer import *
        explainer_clf = RBFLocalExplainer(model)  # use same explainer interface as for regression
        shapley_values_clf = explainer_clf.explain(X_tbx[0])

        ## GEMFIX
        X_bg = shap.sample(X, 100)
        gemfix = GEMFIX(fn, X, lam=0.001)
        gem_values = gemfix.shap_values(X_tbx, nsamples=X_sample_no)
        gem_ranks = create_rank(np.array(gem_values).squeeze())
        gem_avg_ranks = np.mean(gem_ranks[:,feature_imp], axis=1)
        gemfix_mean_rank = np.mean(gem_avg_ranks)

        ## SHAP
        explainer = shap.KernelExplainer(fn, X_bg, l1_reg=False)
        shap_values = explainer.shap_values(X_tbx, nsamples=X_sample_no, l1_reg=False)
        shap_ranks = create_rank(shap_values.squeeze())
        shap_avg_ranks = np.mean(shap_ranks[:,feature_imp], axis=1)
        shap_mean_rank = np.mean(shap_avg_ranks)

        ## Sampling SHAP
        sexplainer = shap.SamplingExplainer(fn, X_bg, l1_reg=False)
        sshap_values = sexplainer.shap_values(X_tbx, nsamples=X_sample_no, l1_reg=False, min_samples_per_feature=1)
        sshap_ranks = create_rank(sshap_values.squeeze())
        sshap_avg_ranks = np.mean(sshap_ranks[:,feature_imp], axis=1)
        sshap_mean_rank = np.mean(sshap_avg_ranks)


        ## Bivariate SHAP
        bishap = Bivariate_KernelExplainer(fn, X_bg)
        bishap_values = bishap.shap_values(X_tbx, nsamples=X_sample_no, l1_reg=True)
        bishap_ranks = create_rank(np.array(bishap_values).squeeze())
        bishap_avg_ranks = np.mean(bishap_ranks[:,feature_imp], axis=1)
        bishap_mean_rank = np.mean(bishap_avg_ranks)


        ## LIME, Unbiased SHAP, and MAPLE 
        lime_exp = lime_tabular.LimeTabularExplainer(X_bg, discretize_continuous=False, mode="regression")
        exp_maple = MAPLE(X, y, X, y)

        lime_values = np.empty_like(X_tbx)
        maple_values = np.empty_like(X_tbx)
        for i in range(X_tbx.shape[0]):
            x = X_tbx[i, ]
        
            ## LIME 
            exp = lime_exp.explain_instance(x, fn, num_samples = X_sample_no)
                
            for tpl in exp.as_list():
                lime_values[i, int(tpl[0])] = tpl[1]

            ## MAPLE
            mpl_exp = exp_maple.explain(x)
            maple_values[i,] = (mpl_exp['coefs'][1:]).squeeze()


        lime_ranks = create_rank(lime_values)
        lime_avg_ranks = np.mean(lime_ranks[:,feature_imp], axis=1)
        lime_mean_rank = np.mean(lime_avg_ranks)

        maple_ranks = create_rank(maple_values)
        maple_avg_ranks = np.mean(maple_ranks[:,feature_imp], axis=1)
        maple_mean_rank = np.mean(maple_avg_ranks)

        #plt.boxplot([shogp_avg_ranks, shap_avg_ranks, bishap_avg_ranks, sshap_avg_ranks, ushap_avg_ranks, lime_avg_ranks, maple_avg_ranks])

        
        method_names = ['Kernel SHAP', 'Sampling SHAP', 'Bivariate SHAP', 'LIME',  'MAPLE']
        all_results = [shap_avg_ranks, sshap_avg_ranks, bishap_avg_ranks, lime_avg_ranks, maple_avg_ranks]

        df = pd.DataFrame(all_results, index=method_names)

        mode = 'a' if results_xsl.exists() else 'w'
        # Load the existing Excel file
        book = load_workbook(results_xsl)
        
        # Remove the sheet if it already exists
        if ds_name in book.sheetnames:
            del book[ds_name]
        
        # Write the DataFrame to a new sheet
        with pd.ExcelWriter(results_xsl, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, index=False, sheet_name=ds_name)

    print("done!")
    

