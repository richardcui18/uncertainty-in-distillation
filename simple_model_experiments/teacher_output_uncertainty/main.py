import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import time
import torch.nn as nn
import torch.optim as optim


np.random.seed(1)
torch.manual_seed(1)

K_REPLICATES = 1000
noise_grid = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0])
do_neural_students = True
neural_epochs = 200 
neural_lr = 0.01


boston = fetch_openml(name="boston", version=1, as_frame=True)

X = boston.data.to_numpy()
y = boston.target.to_numpy().astype(float)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
n_train = X_train.shape[0]
y_train_std = np.std(y_train)

# Fit linear teacher and obtain teacher mean predictions on train / test
teacher_lin = LinearRegression()
teacher_lin.fit(X_train, y_train)
mu_T_train = teacher_lin.predict(X_train)
mu_T_test  = teacher_lin.predict(X_test)


# Fit neural teacher
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

print("Training neural teacher (this uses your SimpleNN and train_nn)...")
torch.manual_seed(0)
neural_teacher = SimpleNN(X_train.shape[1])
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
train_nn(neural_teacher, X_train_t, y_train_t, epochs=neural_epochs, lr=neural_lr)
neural_teacher.eval()
with torch.no_grad():
    mu_T_train_nn = neural_teacher(torch.tensor(X_train, dtype=torch.float32)).squeeze().cpu().numpy()
    mu_T_test_nn  = neural_teacher(torch.tensor(X_test, dtype=torch.float32)).squeeze().cpu().numpy()



def train_linear_student_ols(X_train, y_labels):
    reg = LinearRegression()
    reg.fit(X_train, y_labels)
    return reg

def predict_linear_student(reg, X):
    return reg.predict(X)

def train_neural_student_from_labels(X_train_np, y_labels_np, seed):
    torch.manual_seed(seed)
    model = SimpleNN(X_train_np.shape[1])
    Xt = torch.tensor(X_train_np, dtype=torch.float32)
    yt = torch.tensor(y_labels_np.reshape(-1,1), dtype=torch.float32)
    train_nn(model, Xt, yt, epochs=neural_epochs, lr=neural_lr)
    return model

def predict_neural_student(model, X_np):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_np, dtype=torch.float32)
        preds = model(Xt).squeeze().numpy()
    return preds

# Main experiment
results = {}

for sigma in noise_grid:
    start = time.time()

    train_mses_lin = []
    test_mses_lin  = []
    train_mses_nn = []
    test_mses_nn  = []

    for rep in range(K_REPLICATES):
        seed = 1000 + rep + int(sigma*100)
        np.random.seed(seed)
        torch.manual_seed(seed)

        noisy_labels_train = mu_T_train + np.random.normal(loc=0.0, scale=np.sqrt(sigma)*y_train_std, size=n_train)

        # ---------- Linear student ----------
        lin_student = train_linear_student_ols(X_train, noisy_labels_train)
        pred_train_lin = predict_linear_student(lin_student, X_train)
        pred_test_lin  = predict_linear_student(lin_student, X_test)
        train_mses_lin.append(mean_squared_error(y_train, pred_train_lin))
        test_mses_lin.append(mean_squared_error(y_test, pred_test_lin))

        # ---------- Neural student ----------
        if do_neural_students:
            noisy_labels_train = mu_T_train_nn + np.random.normal(loc=0.0, scale=np.sqrt(sigma)*y_train_std, size=n_train)
            model_nn = train_neural_student_from_labels(X_train, noisy_labels_train, seed=seed)
            preds_train_nn = predict_neural_student(model_nn, X_train)
            preds_test_nn  = predict_neural_student(model_nn, X_test)
            train_mses_nn.append(mean_squared_error(y_train, preds_train_nn))
            test_mses_nn.append(mean_squared_error(y_test, preds_test_nn))

    # aggregate
    results[sigma] = {
        'linear': {
            'train_mean': np.mean(train_mses_lin),
            'train_std' : np.var(train_mses_lin),
            'test_mean' : np.mean(test_mses_lin),
            'test_std'  : np.var(test_mses_lin),
        }
    }
    if do_neural_students:
        results[sigma]['neural'] = {
            'train_mean': np.mean(train_mses_nn),
            'train_std' : np.var(train_mses_nn),
            'test_mean' : np.mean(test_mses_nn),
            'test_std'  : np.var(test_mses_nn),
        }

    elapsed = time.time() - start
    print(f"Finished sweep in {elapsed/60:.2f} minutes. K={K_REPLICATES} replicates per sigma.")

print(results)
