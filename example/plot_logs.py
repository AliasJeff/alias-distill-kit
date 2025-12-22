import re
import ast
from pathlib import Path
import matplotlib.pyplot as plt

# ===================== 配置 =====================
LOG_PATH = Path("example/all.log")

TEACHER_DONE = "Teacher model training completed successfully"
STUDENT_DONE = "Student model distillation completed successfully"

TRAINING_DICT_PATTERN = re.compile(r"\{[^{}]*'loss'[^{}]*\}")
KD_DICT_PATTERN = re.compile(r"\{[^{}]*'loss_kd_logits'[^{}]*\}")


# ===================== 解析函数 =====================
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
    loss_kd_logits = []
    loss_kd_hidden = []
    loss_ce = []

    for m in KD_DICT_PATTERN.finditer(text):
        try:
            d = ast.literal_eval(m.group())
            loss_kd_logits.append(d["loss_kd_logits"])
            loss_kd_hidden.append(d["loss_kd_hidden"])
            loss_ce.append(d["loss_ce"])
        except Exception:
            continue

    return loss_kd_logits, loss_kd_hidden, loss_ce


# ===================== 画图函数 =====================
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

    # Loss
    plt.subplot(3, 1, 1)
    plt.plot(t_ep, t_loss, label="Teacher", linewidth=2)
    plt.plot(s_ep, s_loss, label="Student", linewidth=2)
    plt.ylabel("Loss")
    plt.title("Teacher vs Student (Training Metrics)")
    plt.legend()
    plt.grid(True)

    # Grad norm
    plt.subplot(3, 1, 2)
    plt.plot(t_ep, t_gn, label="Teacher", linewidth=2)
    plt.plot(s_ep, s_gn, label="Student", linewidth=2)
    plt.ylabel("Grad Norm")
    plt.legend()
    plt.grid(True)

    # LR
    plt.subplot(3, 1, 3)
    plt.plot(t_ep, t_lr, label="Teacher", linewidth=2)
    plt.plot(s_ep, s_lr, label="Student", linewidth=2)
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

    steps = list(range(1, len(loss_kd_logits) + 1))

    plt.figure(figsize=(10, 6))

    plt.plot(steps, loss_kd_logits, label="KD Logits Loss", marker="o")
    plt.plot(steps, loss_kd_hidden, label="KD Hidden Loss", marker="o")
    plt.plot(steps, loss_ce, label="CE Loss", marker="o")

    plt.xlabel("Training Step (log order)")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ===================== 主流程 =====================
def main():
    text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")

    teacher_end = text.find(TEACHER_DONE)
    student_end = text.find(STUDENT_DONE)

    if teacher_end == -1 or student_end == -1:
        raise RuntimeError("Teacher / Student completion marker not found")

    teacher_text = text[:teacher_end]
    student_text = text[teacher_end + len(TEACHER_DONE):student_end]

    # -------- Training metrics --------
    t_loss, t_gn, t_lr, t_ep = parse_training_metrics(teacher_text)
    s_loss, s_gn, s_lr, s_ep = parse_training_metrics(student_text)

    plot_training("Teacher Training Metrics", t_loss, t_gn, t_lr, t_ep)
    plot_training("Student Training Metrics", s_loss, s_gn, s_lr, s_ep)
    plot_teacher_student_compare(t_ep, t_loss, t_gn, t_lr, s_ep, s_loss, s_gn, s_lr)

    # -------- KD metrics (Student only) --------
    kd_logits, kd_hidden, kd_ce = parse_kd_metrics(student_text)
    plot_kd_losses("Student KD Loss Trends", kd_logits, kd_hidden, kd_ce)


if __name__ == "__main__":
    main()
