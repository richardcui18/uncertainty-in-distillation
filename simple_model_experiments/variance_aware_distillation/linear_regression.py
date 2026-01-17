import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set seed
np.random.seed(1)


# Load and split data
boston = fetch_openml(name="boston", version=1, as_frame=True)
X = boston.data.to_numpy()
y = boston.target.to_numpy().astype(float)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)


# Train teacher
teacher = LinearRegression()
teacher.fit(X_train, y_train)

mu_T_train = teacher.predict(X_train)
mu_T_test = teacher.predict(X_test)


# Experiment parameters
alphas = [0.01, 0.1, 0.5, 1.0]
ks = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 30, 40, 50]
n_runs = 1000

results_baseline = {k: {"mean": [], "std": []} for k in ks}
results_variance_aware = {k: {"mean": [], "std": []} for k in ks}


# Experiment loop
for alpha in alphas:
    sigma_T2_true = alpha * np.var(y_train)

    for k in ks:
        mse_baseline_runs = []
        mse_varaware_runs = []

        for run in range(n_runs):
            # Generate k noisy teacher outputs
            noisy_teacher_outputs = np.array([
                mu_T_train + np.random.normal(0, np.sqrt(sigma_T2_true), size=len(mu_T_train))
                for _ in range(k)
            ])

            # --- Averaging ---
            baseline_labels = noisy_teacher_outputs.mean(axis=0)
            student_baseline = LinearRegression()
            student_baseline.fit(X_train, baseline_labels)
            mu_S_test = student_baseline.predict(X_test)
            mse_baseline_runs.append(mean_squared_error(y_test, mu_S_test))

            # --- Variance-weighting ---
            # Estimate teacher variance across k samples (per datapoint)
            sigma_T2_est = noisy_teacher_outputs.var(axis=0)

            # Student proxy variance
            sigma_S2 = sigma_T2_true

            # Compute per-point responsibility weights
            w_T = 1 / (sigma_T2_est + 1e-8)  # avoid div by zero
            w_S = 1 / (sigma_S2 + 1e-8)
            alpha_T = w_T / (w_T + w_S)
            alpha_S = w_S / (w_T + w_S)

            # Variance-aware "soft labels" for training
            varaware_labels = alpha_T * noisy_teacher_outputs.mean(axis=0) + alpha_S * mu_T_train

            student_varaware = LinearRegression()
            student_varaware.fit(X_train, varaware_labels)
            mu_S_va_test = student_varaware.predict(X_test)
            mse_varaware_runs.append(mean_squared_error(y_test, mu_S_va_test))

        # Record statistics
        results_baseline[k]["mean"].append(np.mean(mse_baseline_runs))
        results_baseline[k]["std"].append(np.std(mse_baseline_runs))

        results_variance_aware[k]["mean"].append(np.mean(mse_varaware_runs))
        results_variance_aware[k]["std"].append(np.std(mse_varaware_runs))
