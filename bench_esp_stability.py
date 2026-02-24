import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from explainer.LocalExplainer import RBFLocalExplainer

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------


def make_gp_data(n, d, seed=0):
    X, y = make_regression(n_samples=n, n_features=d, random_state=seed)
    return X.astype(np.float64), y.astype(np.float64)


def train_gp(X_train, y_train, n_restarts=2):
    kernel = RBF(1.0, (1e-3, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts, alpha=1e-2)
    gp.fit(X_train, y_train)
    return gp


def run_benchmark(n=500, d_list=None, trials=5, seed=42, gp_restarts=2, esp_methods: list = None, esp_kwargs: dict = None):
    if d_list is None:
        d_list = [10, 15, 20, 25, 30, 40, 50, 70, 80, 100, 200, 500, 1000]

    results = {}

    for d in d_list:
        results[d] = {}
        # Prepare data and train GP
        X, y = make_gp_data(n, d, seed=seed + d)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed + d)
        gp = train_gp(X_train, y_train, n_restarts=gp_restarts)

        # Default ESP methods to test
        if esp_methods is None:
            esp_methods = ["quadratic", "fft", "chebyshev"]

        val_fun = "product"
        # For each ESP method, initialize explainer and run trials
        for method in esp_methods:
            explainer = RBFLocalExplainer(gp, value_function=val_fun, esp_method=method, esp_kwargs=esp_kwargs)

            err_sum = 0.0
            t_sum = 0.0
            count = 0
            failed = False

            for t in range(trials):
                x = X_test[t % len(X_test)]
                y_pred = gp.predict([x])[0]
                try:
                    start = time.time()
                    shap_vals = explainer.explain(x)
                    elapsed = time.time() - start
                    t_sum += elapsed
                    sum_phi = float(np.sum(shap_vals))
                    err = abs(sum_phi - (y_pred - explainer.null_game))
                    err_sum += err
                    count += 1
                except Exception as e:
                    failed = True
                    print(f"[WARN] d={d}, method={method} explainer.explain failed on trial {t}: {e}")
                    break

            if count > 0:
                results[d][method] = {
                    "avg_error": err_sum / count,
                    "avg_time_sec": t_sum / count,
                    "trials": count,
                    "failed": failed,
                }
            else:
                results[d][method] = {
                    "avg_error": None,
                    "avg_time_sec": None,
                    "trials": 0,
                    "failed": True,
                }

    return results


def print_results(results):
    print("\n=== GP LocalExplainer Stability Benchmark (avg abs error & time) ===")
    for d in sorted(results.keys()):
        print(f"\n# Features d={d}")
        for name, stats in results[d].items():
            err = stats["avg_error"]
            t = stats["avg_time_sec"]
            trials = stats["trials"]
            failed = stats["failed"]
            if err is None:
                print(f"- {name}: FAILED (trials={trials})")
            else:
                print(f"- {name}: avg_error={err:.3e}, avg_time={t:.3e}s (trials={trials}){' [failed]' if failed else ''}")


if __name__ == "__main__":
    results = run_benchmark(n=500, trials=5, gp_restarts=0)
    print_results(results)

