import re
import ast
from pathlib import Path
import matplotlib.pyplot as plt

LOG_PATH = Path("example/all.log")

TEACHER_DONE = "Teacher model training completed successfully"
STUDENT_DONE = "Student model distillation completed successfully"

DICT_PATTERN = re.compile(r"\{[^{}]*'loss'[^{}]*\}")


def parse_metrics(text: str):
    losses, grad_norms, lrs, epochs = [], [], [], []

    for m in DICT_PATTERN.finditer(text):
        try:
            d = ast.literal_eval(m.group())
            losses.append(d.get("loss"))
            grad_norms.append(d.get("grad_norm"))
            lrs.append(d.get("learning_rate"))
            epochs.append(d.get("epoch"))
        except Exception:
            continue

    return losses, grad_norms, lrs, epochs


def plot_single(title, losses, grad_norms, lrs, epochs):
    if not losses:
        print(f"[WARN] No data for {title}")
        return

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, losses)
    plt.ylabel("Loss")
    plt.title(title)

    plt.subplot(3, 1, 2)
    plt.plot(epochs, grad_norms)
    plt.ylabel("Grad Norm")

    plt.subplot(3, 1, 3)
    plt.plot(epochs, lrs)
    plt.ylabel("Learning Rate")
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.show()


def plot_teacher_student_compare(t_ep, t_loss, t_gn, t_lr, s_ep, s_loss, s_gn, s_lr):
    if not t_loss or not s_loss:
        print("[WARN] Skip comparison plot (missing data)")
        return

    plt.figure(figsize=(12, 9))

    # -------- Loss --------
    plt.subplot(3, 1, 1)
    plt.plot(t_ep, t_loss, label="Teacher", linewidth=2)
    plt.plot(s_ep, s_loss, label="Student", linewidth=2)
    plt.ylabel("Loss")
    plt.title("Teacher vs Student Comparison")
    plt.legend()
    plt.grid(True)

    # -------- Grad Norm --------
    plt.subplot(3, 1, 2)
    plt.plot(t_ep, t_gn, label="Teacher", linewidth=2)
    plt.plot(s_ep, s_gn, label="Student", linewidth=2)
    plt.ylabel("Grad Norm")
    plt.legend()
    plt.grid(True)

    # -------- Learning Rate --------
    plt.subplot(3, 1, 3)
    plt.plot(t_ep, t_lr, label="Teacher", linewidth=2)
    plt.plot(s_ep, s_lr, label="Student", linewidth=2)
    plt.ylabel("Learning Rate")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")

    # ---------- 找关键点 ----------
    teacher_end = text.find(TEACHER_DONE)
    student_end = text.find(STUDENT_DONE)

    if teacher_end == -1:
        print("[ERROR] Teacher completion marker not found")
        return

    if student_end == -1:
        print("[ERROR] Student completion marker not found")
        return

    # ---------- 严格切分 ----------
    teacher_text = text[:teacher_end]
    student_text = text[teacher_end + len(TEACHER_DONE):student_end]

    # ---------- 解析 ----------
    t_loss, t_gn, t_lr, t_ep = parse_metrics(teacher_text)
    s_loss, s_gn, s_lr, s_ep = parse_metrics(student_text)

    # ---------- 各自画 ----------
    plot_single("Teacher Training Metrics", t_loss, t_gn, t_lr, t_ep)
    plot_single("Student Distillation Metrics", s_loss, s_gn, s_lr, s_ep)

    # ---------- 对比图 ----------
    plot_teacher_student_compare(t_ep, t_loss, t_gn, t_lr, s_ep, s_loss, s_gn, s_lr)


if __name__ == "__main__":
    main()
