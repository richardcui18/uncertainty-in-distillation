import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_digits,
    load_wine,
    load_breast_cancer,
    fetch_covtype,
)
from torchvision.datasets import MNIST

DEFAULT_TEST_SIZE = 0.2
SEED = 1


def _standardize_features(X):
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled, scaler


def load_dataset(
    name: str,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = SEED,
    scale: bool = True,
    covtype_n_samples: int | None = 50000,
):
    name = name.lower()

    if name == "digits":
        ds = load_digits()
        X = ds.data.astype(np.float32)
        y = ds.target.astype(int)
        difficulty = "easy-medium"
        n_classes = len(np.unique(y))

    elif name == "wine":
        ds = load_wine()
        X = ds.data.astype(np.float32)
        y = ds.target.astype(int)
        difficulty = "easy-medium"
        n_classes = len(np.unique(y))

    elif name in ("breast_cancer", "cancer"):
        ds = load_breast_cancer()
        X = ds.data.astype(np.float32)
        y = ds.target.astype(int)
        difficulty = "easy"
        n_classes = len(np.unique(y))

    elif name == "mnist":
        root = "./data"
        train_ds = MNIST(root=root, train=True, download=True)
        test_ds = MNIST(root=root, train=False, download=True)

        X = np.concatenate(
            [train_ds.data.numpy(), test_ds.data.numpy()], axis=0
        )
        y = np.concatenate(
            [train_ds.targets.numpy(), test_ds.targets.numpy()], axis=0
        )

        X = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        y = y.astype(int)

        difficulty = "medium-hard"
        n_classes = len(np.unique(y))

    elif name == "covtype":
        ds = fetch_covtype()
        X = ds["data"].astype(np.float32)
        y = ds["target"].astype(int) - 1
        n_classes = len(np.unique(y))

        if covtype_n_samples is not None and covtype_n_samples < X.shape[0]:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(X.shape[0], size=covtype_n_samples, replace=False)
            X = X[idx]
            y = y[idx]
        difficulty = "hard"

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            "Supported: digits, wine, breast_cancer, mnist, covtype"
        )

    if scale:
        X, scaler = _standardize_features(X)
    else:
        scaler = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    input_dim = X_train.shape[1]
    num_classes = n_classes

    meta_info = {
        "name": name,
        "difficulty": difficulty,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(num_classes),
        "scaled": bool(scale),
        "scaler_type": "StandardScaler" if scale else None,
    }

    return X_train, X_test, y_train, y_test, input_dim, num_classes, meta_info
