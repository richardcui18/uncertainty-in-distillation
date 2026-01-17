import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json


np.random.seed(1)
torch.manual_seed(1)

K_REPLICATES = 10000   # number of independent students per init-noise level
sigma_init_grid = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
neural_epochs = 300
neural_lr = 0.01
weight_decay = 0.0
device = torch.device("cpu")


boston = fetch_openml(name="boston", version=1, as_frame=True)
X_df = boston.data.select_dtypes(include=[np.number]).astype(np.float32)
X = X_df.to_numpy()
y = boston.target.to_numpy().astype(float)


scaler_X = StandardScaler().fit(X)
scaler_y = StandardScaler().fit(y.reshape(-1, 1))
X = scaler_X.transform(X).astype(np.float32)
y = scaler_y.transform(y.reshape(-1, 1)).flatten().astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def collect_param_vector(model):
    """Flatten model params into one numpy vector (CPU)."""
    return torch.cat([p.detach().cpu().view(-1) for p in model.parameters()]).numpy()

def compute_avg_variance_across_students(param_matrix):
    """param_matrix shape: (n_students, num_params) -> average variance across parameter dims"""
    return np.mean(np.var(param_matrix, axis=0))

def make_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.float_)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    return obj

# Model & helpers
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.out(x)
        return x

def kaiming_init_and_perturb(model: nn.Module, sigma_init: float):
    """
    Kaiming init then multiplicative perturbation:
    W <- W * (1 + eps), eps ~ N(0, sigma_init^2)
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if sigma_init > 0.0:
                with torch.no_grad():
                    eps = torch.randn_like(m.weight) * sigma_init
                    m.weight.mul_(1.0 + eps)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                if sigma_init > 0.0:
                    with torch.no_grad():
                        m.bias.add_(torch.randn_like(m.bias) * (sigma_init * 0.1))

def train_nn(model, X_np, y_np, epochs=200, lr=0.01, weight_decay=0.0,
             device=torch.device("cpu"), tracked_param_name=None, storage_list=None):
    """
    Train model and optionally record the first element of a named parameter each epoch.
    If tracked_param_name is None, no recording occurs.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_np.reshape(-1,1), dtype=torch.float32, device=device)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

        if tracked_param_name is not None and storage_list is not None:
            param = dict(model.named_parameters()).get(tracked_param_name, None)
            if param is not None:
                storage_list.append(float(param.detach().cpu().view(-1)[0].item()))
            else:
                storage_list.append(float('nan'))
    return model

def predict_np(model, X_np, device=torch.device("cpu")):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_np, dtype=torch.float32, device=device)
        preds = model(Xt).squeeze().cpu().numpy()   # returns shape (N,)
    return preds

# Tracking setup
before_param_vectors = []
after_param_vectors  = []

tracked_param_name = "fc1.bias"
trajectories_by_sigma = {}
orig_value_by_sigma = {}
noisy_value_by_sigma = {}

# Experiment sweep
results = {}
base_seed = 1234

for sigma_init in sigma_init_grid:
    start_time = time.time()
    test_mses = []

    for rep in range(K_REPLICATES):
        seed = base_seed + rep + int(sigma_init * 1e6)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Build fresh model
        model = SimpleNN(input_dim=X_train.shape[1], hidden_dim=128)

        do_trace_this_sigma = (rep == 0)

        # record original param value for representative (before perturbation)
        if do_trace_this_sigma:
            p0 = dict(model.named_parameters()).get(tracked_param_name, None)
            orig_value_by_sigma[sigma_init] = float(p0.detach().cpu().view(-1)[0].item()) if p0 is not None else None

        # collect full param vector BEFORE perturbation
        before_param_vectors.append(collect_param_vector(model))

        # Apply Kaiming init + multiplicative perturbation
        kaiming_init_and_perturb(model, sigma_init=sigma_init)

        # record noisy param value (immediately after perturbation) for representative
        if do_trace_this_sigma:
            p1 = dict(model.named_parameters()).get(tracked_param_name, None)
            noisy_value_by_sigma[sigma_init] = float(p1.detach().cpu().view(-1)[0].item()) if p1 is not None else None

        # prepare local trajectory storage if tracing
        tracked_values_local = [] if do_trace_this_sigma else None

        # Train
        model = train_nn(model, X_train, y_train, epochs=neural_epochs, lr=neural_lr,
                         weight_decay=weight_decay, device=device,
                         tracked_param_name=(tracked_param_name if do_trace_this_sigma else None),
                         storage_list=tracked_values_local)

        # If we traced this representative student, save its trajectory
        if do_trace_this_sigma:
            trajectories_by_sigma[sigma_init] = tracked_values_local

        # Evaluate on held-out test set
        preds_test = predict_np(model, X_test, device=device)
        test_mse = mean_squared_error(y_test, preds_test)
        test_mses.append(test_mse)

        # collect full param vector AFTER training
        after_param_vectors.append(collect_param_vector(model))

    # aggregate results for this sigma
    test_mses = np.array(test_mses, dtype=np.float64)
    results[sigma_init] = {
        'test_mses_all' : test_mses,
        'test_mean'     : float(test_mses.mean()),
        'test_var'      : float(test_mses.var()),
        'test_std'      : float(test_mses.std()),
        'n_replicates'  : int(len(test_mses))
    }
    elapsed = time.time() - start_time
    print(f"sigma_init={sigma_init:.4f}  -> mean test MSE = {results[sigma_init]['test_mean']:.4f}, "
          f"std = {results[sigma_init]['test_std']:.4f}  (took {elapsed:.1f}s)")

# Summary & Saving
print("\nSummary (sigma_init -> mean ± std of test MSE):")
for s in sigma_init_grid:
    r = results[s]
    print(f"  {s:.4f}  :  {r['test_mean']:.4f}  ± {r['test_std']:.4f}  (n={r['n_replicates']})")

# Compute variance across students (before & after)
before_param_matrix = np.stack(before_param_vectors)   # shape (n_students, n_params)
after_param_matrix  = np.stack(after_param_vectors)

before_var = compute_avg_variance_across_students(before_param_matrix)
after_var  = compute_avg_variance_across_students(after_param_matrix)

print(f"\nAverage variance across students BEFORE training: {before_var:.6e}")
print(f"Average variance across students AFTER  training: {after_var:.6e}")

# Save results summary
out_summary = {
    "results_by_sigma": {str(float(s)): results[s] for s in sigma_init_grid.tolist()},
    "avg_param_variance_before": float(before_var),
    "avg_param_variance_after": float(after_var),
    "tracked_param_name": tracked_param_name,
    "orig_value_by_sigma": {str(float(s)): make_json_safe(orig_value_by_sigma.get(s, None)) for s in orig_value_by_sigma},
    "noisy_value_by_sigma": {str(float(s)): make_json_safe(noisy_value_by_sigma.get(s, None)) for s in noisy_value_by_sigma},
    "n_replicates": int(K_REPLICATES),
}
out_fn = "init_noise_sweep_summary_with_tracking.json"
with open(out_fn, "w") as f:
    json.dump(out_summary, f, indent=2, default=make_json_safe)
print(f"Saved summary to {out_fn}")

# Save trajectories for all sigmas
traj_out = {
    "tracked_param_name": tracked_param_name,
    "trajectories": {str(float(s)): trajectories_by_sigma[s] for s in sorted(trajectories_by_sigma.keys())},
    "orig_values": {str(float(s)): make_json_safe(orig_value_by_sigma.get(s, None)) for s in sorted(orig_value_by_sigma.keys())},
    "noisy_init_values": {str(float(s)): make_json_safe(noisy_value_by_sigma.get(s, None)) for s in sorted(noisy_value_by_sigma.keys())}
}
traj_json_fn = "param_trajectories_all_sigmas.json"
with open(traj_json_fn, "w") as f:
    json.dump(traj_out, f, indent=2, default=make_json_safe)
print(f"Saved all trajectories to {traj_json_fn}")
