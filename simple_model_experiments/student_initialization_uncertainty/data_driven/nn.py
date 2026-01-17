import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim


SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
N_STUDENTS = 1000
TEACHER_EPOCHS = 300
STUDENT_EPOCHS = 300
LR_TEACHER = 1e-3
LR_STUDENT = 1e-3
HIDDEN_TEACHER = 128
HIDDEN_STUDENT = 64

BOOTSTRAP_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# Load and split data
boston = fetch_openml(name="boston", version=1, as_frame=True)
X = boston.data.to_numpy().astype(np.float32)
y = boston.target.to_numpy().astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

scaler_X = StandardScaler().fit(X_train)
X_train = scaler_X.transform(X_train).astype(np.float32)
X_test = scaler_X.transform(X_test).astype(np.float32)

scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel().astype(np.float32)

n_train = X_train.shape[0]
n_test = X_test.shape[0]

print(f"Train size: {n_train}, Test size: {n_test}")


# Model definition
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.out(x)
        return x

def train_nn(model, X_np, y_np_scaled, epochs, lr):
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_np_scaled.reshape(-1, 1), dtype=torch.float32, device=DEVICE)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()
    return model

def predict_nn(model, X_np):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
        preds_scaled = model(X_t).cpu().numpy().ravel()
    return preds_scaled


# Train TEACHER
input_dim = X_train.shape[1]

torch.manual_seed(SEED)
teacher = MLPRegressor(input_dim=input_dim, hidden_dim=HIDDEN_TEACHER)
teacher = train_nn(teacher, X_train, y_train_scaled, epochs=TEACHER_EPOCHS, lr=LR_TEACHER)

mu_T_train_scaled = predict_nn(teacher, X_train)
mu_T_test_scaled = predict_nn(teacher, X_test)

teacher_mse = mean_squared_error(y_test_scaled, mu_T_test_scaled)
print(f"Teacher NN test MSE: {teacher_mse:.4f}")


# Baseline STUDENTS (no bootstrap)

# Variant A baseline: student trained once on teacher outputs
torch.manual_seed(SEED + 1)
student_A_base = MLPRegressor(input_dim=input_dim, hidden_dim=HIDDEN_STUDENT)
student_A_base = train_nn(student_A_base, X_train, mu_T_train_scaled,
                          epochs=STUDENT_EPOCHS, lr=LR_STUDENT)
mu_SA_test_scaled = predict_nn(student_A_base, X_test)
mse_SA_base = mean_squared_error(y_test_scaled, mu_SA_test_scaled)
print(f"Baseline student A (teacher labels) MSE: {mse_SA_base:.4f}")

# Variant B baseline: student trained once on ground truth
torch.manual_seed(SEED + 2)
student_B_base = MLPRegressor(input_dim=input_dim, hidden_dim=HIDDEN_STUDENT)
student_B_base = train_nn(student_B_base, X_train, y_train_scaled,
                          epochs=STUDENT_EPOCHS, lr=LR_STUDENT)
mu_SB_test_scaled = predict_nn(student_B_base, X_test)
mse_SB_base = mean_squared_error(y_test_scaled, mu_SB_test_scaled)
print(f"Baseline student B (ground truth) MSE: {mse_SB_base:.4f}")


# Prepare bootstrap size grid
bootstrap_sizes = sorted(list({max(1, int(frac * n_train)) for frac in BOOTSTRAP_FRACTIONS}))
print(f"\nBootstrap sizes (number of training points per student): {bootstrap_sizes}")

mse_means_A = []
mse_stds_A = []
predvar_means_A = []

mse_means_B = []
mse_stds_B = []
predvar_means_B = []


# Loop over bootstrap sizes
for m in bootstrap_sizes:
    print(f"\n=== Bootstrap size m = {m} ===")

    # ----- Variant A: bootstrap on teacher outputs -----
    mse_boot_A = []
    preds_boot_A = []

    print(f"  Variant A (teacher outputs): training {N_STUDENTS} students...")
    for b in range(N_STUDENTS):
        idx = np.random.randint(0, n_train, size=m)
        X_boot = X_train[idx]
        y_boot_scaled = mu_T_train_scaled[idx]

        torch.manual_seed(1234)
        student = MLPRegressor(input_dim=input_dim, hidden_dim=HIDDEN_STUDENT)
        student = train_nn(student, X_boot, y_boot_scaled,
                           epochs=STUDENT_EPOCHS, lr=LR_STUDENT)

        y_pred_scaled = predict_nn(student, X_test)
        mse_b = mean_squared_error(y_test_scaled, y_pred_scaled)

        mse_boot_A.append(mse_b)
        preds_boot_A.append(y_pred_scaled)

    mse_boot_A = np.array(mse_boot_A)
    preds_boot_A = np.stack(preds_boot_A, axis=0)
    var_preds_A = preds_boot_A.var(axis=0)
    mean_predvar_A = float(var_preds_A.mean())

    mean_mse_A = float(mse_boot_A.mean())
    std_mse_A = float(mse_boot_A.std())

    mse_means_A.append(mean_mse_A)
    mse_stds_A.append(std_mse_A)
    predvar_means_A.append(mean_predvar_A)

    print(f"    A: MSE mean={mean_mse_A:.4f}, std={std_mse_A:.4f}, "
          f"mean pred var={mean_predvar_A:.6e}")

    # ----- Variant B: bootstrap on ground truth -----
    mse_boot_B = []
    preds_boot_B = []

    print(f"  Variant B (ground truth): training {N_STUDENTS} students...")
    for b in range(N_STUDENTS):
        idx = np.random.randint(0, n_train, size=m)
        X_boot = X_train[idx]
        y_boot_scaled = y_train_scaled[idx]

        torch.manual_seed(1234)
        student = MLPRegressor(input_dim=input_dim, hidden_dim=HIDDEN_STUDENT)
        student = train_nn(student, X_boot, y_boot_scaled,
                           epochs=STUDENT_EPOCHS, lr=LR_STUDENT)

        y_pred_scaled = predict_nn(student, X_test)
        mse_b = mean_squared_error(y_test_scaled, y_pred_scaled)

        mse_boot_B.append(mse_b)
        preds_boot_B.append(y_pred_scaled)

    mse_boot_B = np.array(mse_boot_B)
    preds_boot_B = np.stack(preds_boot_B, axis=0)
    var_preds_B = preds_boot_B.var(axis=0)
    mean_predvar_B = float(var_preds_B.mean())

    mean_mse_B = float(mse_boot_B.mean())
    std_mse_B = float(mse_boot_B.std())

    mse_means_B.append(mean_mse_B)
    mse_stds_B.append(std_mse_B)
    predvar_means_B.append(mean_predvar_B)

    print(f"    B: MSE mean={mean_mse_B:.4f}, std={std_mse_B:.4f}, "
          f"mean pred var={mean_predvar_B:.6e}")

# Convert to numpy arrays
bootstrap_sizes = np.array(bootstrap_sizes, dtype=int)
mse_means_A = np.array(mse_means_A)
mse_stds_A = np.array(mse_stds_A)
predvar_means_A = np.array(predvar_means_A)

mse_means_B = np.array(mse_means_B)
mse_stds_B = np.array(mse_stds_B)
predvar_means_B = np.array(predvar_means_B)

bootstrap_fracs = [BOOTSTRAP_FRACTIONS[i]*100 for i in range(len(BOOTSTRAP_FRACTIONS))]
