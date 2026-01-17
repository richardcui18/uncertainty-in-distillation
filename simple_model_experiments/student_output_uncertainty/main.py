import argparse
import json
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
from models import build_model

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)

DEFAULT_TEACHER_EPOCHS = 300
DEFAULT_STUDENT_EPOCHS = 300
DEFAULT_LR_TEACHER = 0.01
DEFAULT_LR_STUDENT = 0.01
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_DEVICE = torch.device("cpu")


def make_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.float_)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    return obj


def predict_proba_and_labels(model, X_np, device=torch.device("cpu")):
    """Return softmax probabilities and predicted labels for X_np."""
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_np, dtype=torch.float32, device=device)
        logits = model(Xt)  # (N, C)
        probs = torch.softmax(logits, dim=1)
        probs_np = probs.cpu().numpy()
        preds_np = np.argmax(probs_np, axis=1)
    return probs_np, preds_np


def compute_group_stats(entropy_vals, preds, true_labels):
    """Compute mean, std, and counts of entropy for correct vs incorrect predictions."""
    correct_mask = preds == true_labels
    incorrect_mask = ~correct_mask

    ent_correct = entropy_vals[correct_mask]
    ent_incorrect = entropy_vals[incorrect_mask]

    if ent_correct.size > 0:
        mean_correct = float(ent_correct.mean())
        std_correct = float(ent_correct.std())
        n_correct = int(ent_correct.size)
    else:
        mean_correct = std_correct = float("nan")
        n_correct = 0

    if ent_incorrect.size > 0:
        mean_incorrect = float(ent_incorrect.mean())
        std_incorrect = float(ent_incorrect.std())
        n_incorrect = int(ent_incorrect.size)
    else:
        mean_incorrect = std_incorrect = float("nan")
        n_incorrect = 0

    return {
        "mean_correct": mean_correct,
        "std_correct": std_correct,
        "n_correct": n_correct,
        "mean_incorrect": mean_incorrect,
        "std_incorrect": std_incorrect,
        "n_incorrect": n_incorrect,
    }


def train_full_batch(
    model,
    X_np,
    y_np,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    verbose: bool = False,
):
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_np, dtype=torch.long, device=device)

    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion_ce(logits, y_t)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch+1}/{epochs} - loss={loss.item():.4f}")

    return model


def distill(
    teacher_model,
    student_model,
    X_train,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    verbose: bool = False,
):
    teacher_model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        teacher_logits = teacher_model(X_t)
        teacher_probs = torch.softmax(teacher_logits, dim=1)
        teacher_pseudo_labels = torch.argmax(teacher_probs, dim=1)

    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay=weight_decay)

    student_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        student_logits = student_model(X_t)
        loss = criterion_ce(student_logits, teacher_pseudo_labels)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  [Student] Epoch {epoch+1}/{epochs} - loss={loss.item():.4f}")

    return student_model


# Plotting
def plot_entropy_bars(
    teacher_entropy,
    student_entropy,
    teacher_preds,
    student_preds,
    y_test,
    teacher_acc,
    student_acc,
    exp_tag: str,
    outname_prefix: str = "teacher_vs_student_entropy_bars_std",
):
    eps = 1e-12

    teacher_stats = compute_group_stats(teacher_entropy, teacher_preds, y_test)
    student_stats = compute_group_stats(student_entropy, student_preds, y_test)

    N_test = len(y_test)

    # Counts
    teacher_n_correct = teacher_stats["n_correct"]
    teacher_n_incorrect = teacher_stats["n_incorrect"]
    student_n_correct = student_stats["n_correct"]
    student_n_incorrect = student_stats["n_incorrect"]

    labels = [
        f"All (N={N_test})",
        f"Correct (T={teacher_n_correct}, S={student_n_correct})",
        f"Incorrect (T={teacher_n_incorrect}, S={student_n_incorrect})",
    ]

    x = np.arange(len(labels))
    width = 0.35

    # Means
    teacher_mean_entropy_all = float(teacher_entropy.mean())
    student_mean_entropy_all = float(student_entropy.mean())

    teacher_means = np.array(
        [
            teacher_mean_entropy_all,
            teacher_stats["mean_correct"],
            teacher_stats["mean_incorrect"],
        ],
        dtype=float,
    )
    student_means = np.array(
        [
            student_mean_entropy_all,
            student_stats["mean_correct"],
            student_stats["mean_incorrect"],
        ],
        dtype=float,
    )

    # Standard deviations
    teacher_std_entropy_all = float(teacher_entropy.std())
    student_std_entropy_all = float(student_entropy.std())

    teacher_stds = np.array(
        [
            teacher_std_entropy_all,
            teacher_stats["std_correct"],
            teacher_stats["std_incorrect"],
        ],
        dtype=float,
    )
    student_stds = np.array(
        [
            student_std_entropy_all,
            student_stats["std_correct"],
            student_stats["std_incorrect"],
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    teacher_color = "mediumseagreen"
    student_color = "#4da6ff"

    ax.bar(
        x - width / 2,
        teacher_means,
        width,
        yerr=teacher_stds,
        capsize=5,
        label="Teacher",
        color=teacher_color,
        alpha=0.9,
    )
    ax.bar(
        x + width / 2,
        student_means,
        width,
        yerr=student_stds,
        capsize=5,
        label="Student",
        color=student_color,
        alpha=0.9,
    )

    teacher_acc_pct = teacher_acc * 100.0
    student_acc_pct = student_acc * 100.0

    ax.set_title(
        f"Teacher vs Student Predictive Entropy ({exp_tag})\n"
        f"Teacher Acc = {teacher_acc_pct:.2f}%  |  Student Acc = {student_acc_pct:.2f}%",
        fontsize=18,
        pad=18,
    )

    ax.set_xlabel("Prediction Group", fontsize=16)
    ax.set_ylabel("Predictive Entropy", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(fontsize=13, loc="upper left")
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.6)

    plt.tight_layout()
    outname = f"{outname_prefix}_{exp_tag}.png"
    plt.savefig(outname, dpi=300, bbox_inches="tight")
    print(f"Saved bar plot to {outname}")
    plt.show()


# ------------------- Main -------------------
def main():
    parser = argparse.ArgumentParser(
        description="Teacher-Student uncertainty with knowledge distillation."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="digits",
        help="Dataset name: digits, wine, breast_cancer, mnist, covtype",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="nn",
        help="Teacher model type: nn, logreg",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="nn_small",
        help="Student model type: nn_small, nn, logreg",
    )
    parser.add_argument("--teacher_epochs", type=int, default=DEFAULT_TEACHER_EPOCHS)
    parser.add_argument("--student_epochs", type=int, default=DEFAULT_STUDENT_EPOCHS)
    parser.add_argument("--lr_teacher", type=float, default=DEFAULT_LR_TEACHER)
    parser.add_argument("--lr_student", type=float, default=DEFAULT_LR_STUDENT)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda (if available)",
    )
    parser.add_argument(
        "--covtype_n_samples",
        type=int,
        default=50000,
        help="Subsample size for covtype (None = use all).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print training progress.",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Load dataset
    print(f"Loading dataset '{args.dataset}'...")
    X_train, X_test, y_train, y_test, input_dim, num_classes, meta = load_dataset(
        args.dataset,
        test_size=0.2,
        random_state=SEED,
        scale=True,
        covtype_n_samples=args.covtype_n_samples,
    )
    print(
        f"Dataset '{meta['name']}': n_samples={meta['n_samples']}, "
        f"n_features={meta['n_features']}, n_classes={meta['n_classes']}, "
        f"difficulty={meta['difficulty']}"
    )

    # 2. Build TEACHER
    print(
        f"\nTraining TEACHER model "
        f"({args.teacher_model}) on ground truth labels..."
    )
    teacher = build_model(
        model_type=args.teacher_model,
        role="teacher",
        input_dim=input_dim,
        num_classes=num_classes,
        device=device,
    )

    start_time = time.time()
    teacher = train_full_batch(
        teacher,
        X_train,
        y_train,
        epochs=args.teacher_epochs,
        lr=args.lr_teacher,
        weight_decay=args.weight_decay,
        device=device,
        verbose=args.verbose,
    )
    elapsed = time.time() - start_time
    print(f"Teacher training completed in {elapsed:.1f}s.")

    # 3. Teacher eval
    eps = 1e-12
    teacher_probs_test, teacher_preds_test = predict_proba_and_labels(
        teacher, X_test, device=device
    )
    teacher_acc = (teacher_preds_test == y_test).mean()
    print(f"Teacher test accuracy: {teacher_acc:.4f}")

    teacher_entropy = -np.sum(
        teacher_probs_test * np.log(teacher_probs_test + eps), axis=1
    )
    teacher_mean_entropy_all = float(teacher_entropy.mean())
    teacher_median_entropy_all = float(np.median(teacher_entropy))
    teacher_std_entropy_all = float(teacher_entropy.std())

    print(f"Teacher mean predictive entropy (all):   {teacher_mean_entropy_all:.4f}")
    print(f"Teacher median predictive entropy (all): {teacher_median_entropy_all:.4f}")

    # 4. Train STUDENT via pure distillation
    print(
        f"\nTraining STUDENT model "
        f"({args.student_model}) using ONLY teacher-provided labels (pure distillation)..."
    )
    student = build_model(
        model_type=args.student_model,
        role="student",
        input_dim=input_dim,
        num_classes=num_classes,
        device=device,
    )

    student = distill(
        teacher,
        student,
        X_train,
        epochs=args.student_epochs,
        lr=args.lr_student,
        weight_decay=args.weight_decay,
        device=device,
        verbose=args.verbose,
    )
    print("Student training completed.")

    # 5. Student eval
    student_probs_test, student_preds_test = predict_proba_and_labels(
        student, X_test, device=device
    )
    student_acc = (student_preds_test == y_test).mean()

    student_entropy = -np.sum(
        student_probs_test * np.log(student_probs_test + eps), axis=1
    )
    student_mean_entropy_all = float(student_entropy.mean())
    student_median_entropy_all = float(np.median(student_entropy))
    student_std_entropy_all = float(student_entropy.std())

    print("\n=== Student evaluation on test set (pure distillation) ===")
    print(f"Student test accuracy: {student_acc:.4f}")
    print(f"Student mean predictive entropy (all):   {student_mean_entropy_all:.4f}")
    print(f"Student median predictive entropy (all): {student_median_entropy_all:.4f}")

    # 6. Compute per-model stats: correct vs incorrect
    teacher_stats = compute_group_stats(teacher_entropy, teacher_preds_test, y_test)
    student_stats = compute_group_stats(student_entropy, student_preds_test, y_test)

    print("\n=== Entropy stats (nats, mean ± std) ===")
    print(
        f"Teacher - correct:   mean={teacher_stats['mean_correct']:.4f}, "
        f"std={teacher_stats['std_correct']:.4f}, n={teacher_stats['n_correct']}"
    )
    print(
        f"Teacher - incorrect: mean={teacher_stats['mean_incorrect']:.4f}, "
        f"std={teacher_stats['std_incorrect']:.4f}, n={teacher_stats['n_incorrect']}"
    )
    print(
        f"Student - correct:   mean={student_stats['mean_correct']:.4f}, "
        f"std={student_stats['std_correct']:.4f}, n={student_stats['n_correct']}"
    )
    print(
        f"Student - incorrect: mean={student_stats['mean_incorrect']:.4f}, "
        f"std={student_stats['std_incorrect']:.4f}, n={student_stats['n_incorrect']}"
    )

    # 7. Save summary
    exp_tag = "Neural Network"
    summary = {
        "dataset": meta,
        "teacher_model_type": args.teacher_model,
        "student_model_type": args.student_model,
        "teacher_test_accuracy": make_json_safe(teacher_acc),
        "teacher_mean_entropy_all": make_json_safe(teacher_mean_entropy_all),
        "teacher_median_entropy_all": make_json_safe(teacher_median_entropy_all),
        "teacher_std_entropy_all": make_json_safe(teacher_std_entropy_all),
        "teacher_entropy_stats": teacher_stats,
        "student_test_accuracy": make_json_safe(student_acc),
        "student_mean_entropy_all": make_json_safe(student_mean_entropy_all),
        "student_median_entropy_all": make_json_safe(student_median_entropy_all),
        "student_std_entropy_all": make_json_safe(student_std_entropy_all),
        "student_entropy_stats": student_stats,
        "teacher_epochs": args.teacher_epochs,
        "student_epochs": args.student_epochs,
        "lr_teacher": args.lr_teacher,
        "lr_student": args.lr_student,
        "weight_decay": args.weight_decay,
    }

    out_fn = f"teacher_student_uncertainty_{exp_tag}.json"
    with open(out_fn, "w") as f:
        json.dump(summary, f, indent=2, default=make_json_safe)
    print(f"\nSaved summary to {out_fn}")

    # 8. Plot
    plot_entropy_bars(
        teacher_entropy,
        student_entropy,
        teacher_preds_test,
        student_preds_test,
        y_test,
        teacher_acc,
        student_acc,
        exp_tag=exp_tag,
    )


if __name__ == "__main__":
    main()
