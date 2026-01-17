import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    """
    Teacher network: one hidden layer MLP.
    """

    def __init__(self, input_dim, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.out = nn.Linear(hidden_dim, output_dim)  # logits

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.out(x)
        return x


class StudentNN(nn.Module):
    """
    Smaller student network: one hidden layer with fewer units.
    """

    def __init__(self, input_dim, hidden_dim=64, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.out(x)
        return x


class LogisticRegressionModel(nn.Module):
    """
    Multinomial logistic regression (softmax regression).
    Equivalent to a single linear layer + softmax used with CrossEntropyLoss.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # No activation here; CrossEntropyLoss expects raw logits
        return self.linear(x)


def build_model(
    model_type: str,
    role: str,
    input_dim: int,
    num_classes: int,
    device: torch.device,
):
    """
    Factory function to build teacher/student models by name.

    Args:
        model_type: string, e.g., "nn", "nn_small", "logreg"
        role: "teacher" or "student" (used to pick sizes when needed)
        input_dim: feature dimensionality
        num_classes: number of classes
        device: torch.device

    Returns:
        nn.Module on the proper device
    """
    model_type = model_type.lower()

    if model_type == "nn":
        # Teacher-style NN
        model = SimpleNN(input_dim=input_dim, hidden_dim=128, output_dim=num_classes)
    elif model_type in ("nn_small", "student_nn"):
        # Student-style NN
        model = StudentNN(input_dim=input_dim, hidden_dim=64, output_dim=num_classes)
    elif model_type in ("logreg", "logistic_regression", "linear"):
        model = LogisticRegressionModel(input_dim=input_dim, output_dim=num_classes)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Supported: 'nn', 'nn_small', 'student_nn', 'logreg'"
        )

    return model.to(device)
