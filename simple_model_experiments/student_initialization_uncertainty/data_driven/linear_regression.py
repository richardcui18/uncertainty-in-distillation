import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

np.random.seed(1)

N_STUDENTS = 1000  # number of bootstrap student models per (variant, bootstrap size)
BOOTSTRAP_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # fractions of n_train for bootstrap size

# Load and split data
boston = fetch_openml(name="boston", version=1, as_frame=True)
X = boston.data.to_numpy()
y = boston.target.to_numpy().astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

n_train = X_train.shape[0]
n_test = X_test.shape[0]

print(f"Train size: {n_train}, Test size: {n_test}")

# Train teacher on ground truth
teacher = LinearRegression()
teacher.fit(X_train, y_train_scaled)

mu_T_train_scaled = teacher.predict(X_train)
mu_T_test_scaled = teacher.predict(X_test)

teacher_mse_test = mean_squared_error(y_test_scaled, mu_T_test_scaled)
print(f"Teacher test MSE: {teacher_mse_test:.4f}")


# Baseline single students (no bootstrap)
# Variant A baseline: student trained once on teacher outputs
student_A_baseline = LinearRegression()
student_A_baseline.fit(X_train, mu_T_train_scaled)
mu_SA_test_scaled = student_A_baseline.predict(X_test)
mse_SA_baseline = mean_squared_error(y_test_scaled, mu_SA_test_scaled)
print(f"Baseline student test MSE: {mse_SA_baseline:.4f}")

# Variant B baseline: student trained once on ground truth
student_B_baseline = LinearRegression()
student_B_baseline.fit(X_train, y_train_scaled)
mu_SB_test_scaled = student_B_baseline.predict(X_test)
mse_SB_baseline = mean_squared_error(y_test_scaled, mu_SB_test_scaled)
print(f"Baseline student test MSE: {mse_SB_baseline:.4f}")


# Prepare bootstrap size grid
bootstrap_sizes = sorted(list({max(1, int(frac * n_train)) for frac in BOOTSTRAP_FRACTIONS}))
print(f"\nBootstrap sizes (number of training points per student): {bootstrap_sizes}")

# To store results
mse_means_A = []
mse_stds_A = []
predvar_means_A = []

mse_means_B = []
mse_stds_B = []
predvar_means_B = []


# Loop over bootstrap sizes
for m in bootstrap_sizes:
    print(f"\n=== Bootstrap size m = {m} ===")

    # Variant A: bootstrap on teacher outputs (scaled y)
    mse_bootstrap_A = []
    preds_bootstrap_A = []

    print(f"  Variant A (teacher outputs): training {N_STUDENTS} students...")
    for b in range(N_STUDENTS):
        idx = np.random.randint(0, n_train, size=m)
        X_train_boot = X_train[idx]
        y_train_boot_scaled = mu_T_train_scaled[idx]

        student = LinearRegression()
        student.fit(X_train_boot, y_train_boot_scaled)

        y_pred_test_scaled = student.predict(X_test)
        mse_b = mean_squared_error(y_test_scaled, y_pred_test_scaled)

        mse_bootstrap_A.append(mse_b)
        preds_bootstrap_A.append(y_pred_test_scaled)

    mse_bootstrap_A = np.array(mse_bootstrap_A)
    preds_bootstrap_A = np.stack(preds_bootstrap_A, axis=0)

    mean_mse_A = float(mse_bootstrap_A.mean())
    std_mse_A = float(mse_bootstrap_A.std())
    var_preds_A = preds_bootstrap_A.var(axis=0)
    mean_predvar_A = float(var_preds_A.mean())

    mse_means_A.append(mean_mse_A)
    mse_stds_A.append(std_mse_A)
    predvar_means_A.append(mean_predvar_A)

    print(f"    A: MSE mean={mean_mse_A:.4f}, std={std_mse_A:.4f}, "
          f"mean predictive var={mean_predvar_A:.6e}")

    # Variant B: bootstrap on ground truth
    mse_bootstrap_B = []
    preds_bootstrap_B = []

    print(f"  Variant B (ground truth): training {N_STUDENTS} students...")
    for b in range(N_STUDENTS):
        idx = np.random.randint(0, n_train, size=m)
        X_train_boot = X_train[idx]
        y_train_boot_scaled = y_train_scaled[idx]

        student = LinearRegression()
        student.fit(X_train_boot, y_train_boot_scaled)

        y_pred_test_scaled = student.predict(X_test)
        mse_b = mean_squared_error(y_test_scaled, y_pred_test_scaled)

        mse_bootstrap_B.append(mse_b)
        preds_bootstrap_B.append(y_pred_test_scaled)

    mse_bootstrap_B = np.array(mse_bootstrap_B)
    preds_bootstrap_B = np.stack(preds_bootstrap_B, axis=0)

    mean_mse_B = float(mse_bootstrap_B.mean())
    std_mse_B = float(mse_bootstrap_B.std())
    var_preds_B = preds_bootstrap_B.var(axis=0)
    mean_predvar_B = float(var_preds_B.mean())

    mse_means_B.append(mean_mse_B)
    mse_stds_B.append(std_mse_B)
    predvar_means_B.append(mean_predvar_B)

    print(f"    B: MSE mean={mean_mse_B:.4f}, std={std_mse_B:.4f}, "
          f"mean predictive var={mean_predvar_B:.6e}")

# Convert to numpy arrays
bootstrap_sizes = np.array(bootstrap_sizes, dtype=int)
mse_means_A = np.array(mse_means_A)
mse_stds_A = np.array(mse_stds_A)
predvar_means_A = np.array(predvar_means_A)

mse_means_B = np.array(mse_means_B)
mse_stds_B = np.array(mse_stds_B)
predvar_means_B = np.array(predvar_means_B)

bootstrap_fracs = [BOOTSTRAP_FRACTIONS[i]*100 for i in range(len(BOOTSTRAP_FRACTIONS))]
