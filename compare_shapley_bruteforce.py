import time
import signal
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from explainer.LocalExplainer import RBFLocalExplainer
from explainer.esp import ESPComputer
import json
import os

plt.rcParams.update({
    'font.size': 14,     # Set default font size
    'font.weight': 'bold',  # Set default font weight to bold
    'axes.labelweight': 'bold',  # Ensure the axis labels are bold
    'axes.titleweight': 'bold',  # Ensure the titles are bold
    'figure.titleweight': 'bold',  # Bold for suptitle if you use fig.suptitle()
    'xtick.labelsize': 14,  # Font size for X-tick labels
    'ytick.labelsize': 14,  # Font size for Y-tick labels
    'xtick.major.size': 5,  # Length of major ticks
    'ytick.major.size': 5,  # Length of major ticks
    'xtick.minor.size': 3,  # Length of minor ticks
    'ytick.minor.size': 3,   # Length of minor ticks
    'figure.dpi': 200,
    'savefig.dpi': 800,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'text.antialiased': True,
    'lines.antialiased': True,
    'axes.linewidth': 1.2
})



def load_results_if_plot_only(results_path):
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    return None


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


def run_comparison(
    sample_size=1000,
    feature_sizes=None,
    budget_seconds=100.0,
    random_state=42,
):
    if feature_sizes is None:
        feature_sizes = [1000]#5, 10, 15, 20, 25, 30, 40, 50, 100, 200, 500, 1000]

    results = {
        "feature_size": [],
        "time_fast": [],
        "time_brute": [],
        "max_abs_err": [],
        "max_rel_err": [],
    }

    for d in feature_sizes:
        X, y = make_regression(
            n_samples=sample_size,
            n_features=d,
            noise=0.1,
            random_state=random_state,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        svr = SVR(kernel="rbf")
        svr.fit(X_train, y_train)

        explainer = RBFLocalExplainer(svr)
        x = X_test[0]

        kernel_vectors = explainer.compute_kernel_vectors(explainer.X_train, x).T

        start = time.perf_counter()
        esp = ESPComputer(method="quadratic")
        Omega = esp.compute_weight_vectors(kernel_vectors)
        fast_shap = np.array(
            [
                explainer.alpha.dot((kernel_vectors[:, j] - 1) * Omega[:, j])
                for j in range(d)
            ]
        )
        time_fast = time.perf_counter() - start
        if time_fast > budget_seconds:
            time_fast = float("inf")

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(max(1, budget_seconds)))
        brute_shap = None
        time_brute = float("inf")
        try:
            start = time.perf_counter()
            brute_shap = explainer.brute_force_shapley(kernel_vectors)
            time_brute = time.perf_counter() - start
            signal.alarm(0)
        except TimeoutError:
            signal.alarm(0)

        if time_brute > budget_seconds:
            time_brute = float("inf")
            brute_shap = None

        if brute_shap is not None and np.isfinite(time_fast):
            abs_err = np.abs(fast_shap - brute_shap)
            rel_err = abs_err / np.maximum(np.abs(brute_shap), 1e-12)
            max_abs_err = abs_err.max()
            max_rel_err = rel_err.max()
        else:
            max_abs_err = float("inf")
            max_rel_err = float("inf")

        results["feature_size"].append(d)
        results["time_fast"].append(time_fast)
        results["time_brute"].append(time_brute)
        results["max_abs_err"].append(max_abs_err)
        results["max_rel_err"].append(max_rel_err)

    return results


def plot_results(results, output_path="results/pkexshapley_bruteforce.png"):
    feature_size = np.array(results["feature_size"])
    time_fast = np.array(results["time_fast"])
    time_brute = np.array(results["time_brute"])

    plt.figure(figsize=(7, 3))
    mask_fast = np.isfinite(time_fast)
    mask_brute = np.isfinite(time_brute)
    plt.plot(
        feature_size[mask_fast],
        time_fast[mask_fast],
        marker="o",
        linewidth=3.5,
        label="PKeX-Shapley",
    )
    plt.plot(
        feature_size[mask_brute],
        time_brute[mask_brute],
        marker="o",
        linewidth=3.5,
        label="Brute force",
    )
    plt.xlabel("Number of features")
    plt.ylabel("Time (seconds)")
    plt.xscale("log")
    plt.yscale("log")
    # plt.title("Feature size vs computation time")
    plt.ylim([0,40])
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot to: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_only = True
    results_path = "results/benchmark_results.json"
    
    if plot_only:
        results = load_results_if_plot_only(results_path)

    else:
        results = run_comparison(sample_size=1000, budget_seconds=25.0)
    
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        else:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    plot_results(results)

    print("Results summary:")
    for i, d in enumerate(results["feature_size"]):
        print(
            f"d={d:>3} | fast={results['time_fast'][i]:.4f}s | "
            f"brute={results['time_brute'][i]:.4f}s | "
            f"max_abs_err={results['max_abs_err'][i]:.3e} | "
            f"max_rel_err={results['max_rel_err'][i]:.3e}"
        )
