import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim


# Set seed
np.random.seed(1)
torch.manual_seed(1)


# Load and preprocess data
boston = fetch_openml(name="boston", version=1, as_frame=True)
X_df = boston.data.select_dtypes(include=[np.number]).astype(np.float32)
X = X_df.to_numpy()
y = boston.target.to_numpy().astype(float)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Define simple NN
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_dim, 16)
        self.act = nn.ReLU()
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        return self.out(self.act(self.hidden(x)))

def train_nn(model, X, y, epochs=200, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()


# Train teacher model
teacher = SimpleNN(X_train.shape[1])
train_nn(teacher, X_train_t, y_train_t)
with torch.no_grad():
    mu_T_train = teacher(X_train_t).numpy().flatten()
    mu_T_test = teacher(X_test_t).numpy().flatten()


# Experiment parameters
alphas = [0.01, 0.1, 0.5, 1.0]
ks = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 30, 40, 50]
n_runs = 1000

results_baseline = {k: {"mean": [], "std": []} for k in ks}
results_variance_aware = {k: {"mean": [], "std": []} for k in ks}

alpha_target = 0.5
preds_baseline_per_k = {k: [] for k in ks}
preds_varaware_per_k = {k: [] for k in ks}


# Experiment loop
for alpha in alphas:
    sigma_T2_true = alpha * np.var(y_train)

    for k in ks:
        mse_baseline_runs = []
        mse_varaware_runs = []

        if alpha == alpha_target:
            preds_baseline_per_k[k] = []
            preds_varaware_per_k[k] = []

        for run in range(n_runs):

            # --- Generate noisy teacher outputs ---
            noisy_teacher_outputs = np.array([
                mu_T_train + np.random.normal(0, np.sqrt(sigma_T2_true), size=len(mu_T_train))
                for _ in range(k)
            ])

            # --- Averaging ---
            baseline_labels = noisy_teacher_outputs.mean(axis=0)
            student_baseline = SimpleNN(X_train.shape[1])
            student_train_labels_t = torch.tensor(baseline_labels, dtype=torch.float32).view(-1, 1)
            train_nn(student_baseline, X_train_t, student_train_labels_t)
            with torch.no_grad():
                mu_S_test = student_baseline(X_test_t).numpy().flatten()
            mse_baseline_runs.append(mean_squared_error(y_test, mu_S_test))

            if alpha == alpha_target:
                preds_baseline_per_k[k].append(mu_S_test.copy())

            # --- Variance-weighting ---
            sigma_T2_est = noisy_teacher_outputs.var(axis=0, ddof=0)
            sigma_S2 = sigma_T2_true

            w_T = 1 / (sigma_T2_est + 1e-8)
            w_S = 1 / (sigma_S2 + 1e-8)
            alpha_T = w_T / (w_T + w_S)
            alpha_S = w_S / (w_T + w_S)

            varaware_labels = alpha_T * noisy_teacher_outputs.mean(axis=0) + alpha_S * mu_T_train

            student_varaware = SimpleNN(X_train.shape[1])
            varaware_labels_t = torch.tensor(varaware_labels, dtype=torch.float32).view(-1, 1)
            train_nn(student_varaware, X_train_t, varaware_labels_t)
            with torch.no_grad():
                mu_S_va_test = student_varaware(X_test_t).numpy().flatten()
            mse_varaware_runs.append(mean_squared_error(y_test, mu_S_va_test))

            if alpha == alpha_target:
                preds_varaware_per_k[k].append(mu_S_va_test.copy())

        # Record results
        results_baseline[k]["mean"].append(np.mean(mse_baseline_runs))
        results_baseline[k]["std"].append(np.std(mse_baseline_runs))
        results_variance_aware[k]["mean"].append(np.mean(mse_varaware_runs))
        results_variance_aware[k]["std"].append(np.std(mse_varaware_runs))
