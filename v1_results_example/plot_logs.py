import re
import ast
import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
#                               CONFIG
# =============================================================================

BASE_DIR = Path("example")
LOG_PATH = BASE_DIR / "all.log"

TEACHER_DONE = "Teacher model training completed successfully"
STUDENT_DONE = "Student model distillation completed successfully"

TRAINING_DICT_PATTERN = re.compile(r"\{[^{}]*'loss'[^{}]*\}")
KD_DICT_PATTERN = re.compile(r"\{[^{}]*'loss_kd_logits'[^{}]*\}")

EVAL_JSON_PREFIX = "evaluation_results_"

METRICS = [
    "perplexity",
    "average_loss",
    "bleu_score",
    "f1_score",
    "rouge1",
    "rouge2",
    "rougeL",
]

MODEL_KEYS = [
    "teacher_origin",
    "teacher",
    "original_student",
    "distilled_student",
]

# =============================================================================
#                        LOG PARSING (TRAIN / KD)
# =============================================================================


def parse_training_metrics(text: str):
    losses, grad_norms, lrs, epochs = [], [], [], []

    for m in TRAINING_DICT_PATTERN.finditer(text):
        try:
            d = ast.literal_eval(m.group())
            losses.append(d.get("loss"))
            grad_norms.append(d.get("grad_norm"))
            lrs.append(d.get("learning_rate"))
            epochs.append(d.get("epoch"))
        except Exception:
            continue

    return losses, grad_norms, lrs, epochs


def parse_kd_metrics(text: str):
    loss_kd_logits, loss_kd_hidden, loss_ce = [], [], []

    for m in KD_DICT_PATTERN.finditer(text):
        try:
            d = ast.literal_eval(m.group())
            loss_kd_logits.append(d["loss_kd_logits"])
            loss_kd_hidden.append(d["loss_kd_hidden"])
            loss_ce.append(d["loss_ce"])
        except Exception:
            continue

    return loss_kd_logits, loss_kd_hidden, loss_ce


# =============================================================================
#                                 PLOTS
# =============================================================================


def plot_training(title, losses, grad_norms, lrs, epochs):
    if not losses:
        print(f"[WARN] No training data for {title}")
        return

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, losses)
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(epochs, grad_norms)
    plt.ylabel("Grad Norm")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(epochs, lrs)
    plt.ylabel("Learning Rate")
    plt.xlabel("Epoch")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_teacher_student_compare(t_ep, t_loss, t_gn, t_lr, s_ep, s_loss, s_gn, s_lr):
    if not t_loss or not s_loss:
        print("[WARN] Skip teacher/student comparison (missing data)")
        return

    plt.figure(figsize=(12, 9))

    plt.subplot(3, 1, 1)
    plt.plot(t_ep, t_loss, label="Teacher")
    plt.plot(s_ep, s_loss, label="Student")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t_ep, t_gn, label="Teacher")
    plt.plot(s_ep, s_gn, label="Student")
    plt.ylabel("Grad Norm")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t_ep, t_lr, label="Teacher")
    plt.plot(s_ep, s_lr, label="Student")
    plt.ylabel("Learning Rate")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_kd_losses(title, loss_kd_logits, loss_kd_hidden, loss_ce):
    if not loss_kd_logits:
        print(f"[WARN] No KD loss data for {title}")
        return

    steps = range(1, len(loss_kd_logits) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss_kd_logits, label="KD Logits")
    plt.plot(steps, loss_kd_hidden, label="KD Hidden")
    plt.plot(steps, loss_ce, label="CE")

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# =============================================================================
#                    EVALUATION JSON (LATEST FILE)
# =============================================================================


def find_latest_eval_json():
    files = list(BASE_DIR.glob(f"{EVAL_JSON_PREFIX}*.json"))
    if not files:
        raise FileNotFoundError("No evaluation_results_*.json found")

    def extract_ts(p: Path):
        ts = p.stem.replace(EVAL_JSON_PREFIX, "")
        return datetime.strptime(ts, "%Y%m%d_%H%M%S")

    latest = max(files, key=extract_ts)
    print(f"[INFO] Using latest evaluation file: {latest.name}")
    return latest


def load_eval_data():
    path = find_latest_eval_json()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["models"]


# =============================================================================
#                       EVALUATION PLOTS
# =============================================================================


def annotate_bars(bars, precision=2):
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,
                 h,
                 f"{h:.{precision}f}",
                 ha="center",
                 va="bottom",
                 fontsize=9)


def plot_student_before_after(models):
    before = models["original_student"]
    after = models["distilled_student"]

    metrics = METRICS
    before_vals = [before[m] for m in metrics]
    after_vals = [after[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width / 2, before_vals, width, label="Original Student")
    bars2 = plt.bar(x + width / 2, after_vals, width, label="Distilled Student")

    annotate_bars(bars1)
    annotate_bars(bars2)

    plt.xticks(x, [m.replace("_", "\n") for m in metrics])
    plt.ylabel("Score")
    plt.title("Student Before vs After Distillation")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


# =============================================================================
#                                   MAIN
# =============================================================================


def main():
    # ================= TRAIN / KD LOGS =================
    text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")

    t_end = text.find(TEACHER_DONE)
    s_end = text.find(STUDENT_DONE)

    if t_end == -1 or s_end == -1:
        raise RuntimeError("Teacher / Student completion marker not found")

    teacher_text = text[:t_end]
    student_text = text[t_end + len(TEACHER_DONE):s_end]

    t_loss, t_gn, t_lr, t_ep = parse_training_metrics(teacher_text)
    s_loss, s_gn, s_lr, s_ep = parse_training_metrics(student_text)

    plot_training("Teacher Training", t_loss, t_gn, t_lr, t_ep)
    plot_training("Student Training", s_loss, s_gn, s_lr, s_ep)
    plot_teacher_student_compare(t_ep, t_loss, t_gn, t_lr, s_ep, s_loss, s_gn, s_lr)

    kd_logits, kd_hidden, kd_ce = parse_kd_metrics(student_text)
    plot_kd_losses("Student KD Loss", kd_logits, kd_hidden, kd_ce)

    # ================= EVALUATION =================
    models = load_eval_data()
    plot_student_before_after(models)


if __name__ == "__main__":
    main()
