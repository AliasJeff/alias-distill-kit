import json
import re
import matplotlib.pyplot as plt
from pathlib import Path

DISTILL_FILE = Path("distill.jsonl")
TEACHER_FILE = Path("teacher_training.jsonl")
MAX_POINTS = 1000


def read_distill_log(file_path):  # noqa: C901
    data = {"step": [], "epoch": []}

    if not file_path.exists():
        print(f"[Warning] File not found: {file_path}")
        return None

    print(f"Reading {file_path}...")

    pattern = re.compile(r"(\d+)_(.+)")

    valid_count = 0
    tracked_loss_keys = None

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)

                if "step" not in entry or "epoch" not in entry:
                    continue

                s = int(entry["step"])
                e = float(entry["epoch"])

                temp_losses = {}

                for key, value in entry.items():
                    match = pattern.search(key)
                    if match:
                        loss_name = match.group(2)
                        data_key = f"loss_{loss_name}"
                        temp_losses[data_key] = float(value)

                if not temp_losses:
                    continue

                if tracked_loss_keys is None:
                    tracked_loss_keys = list(temp_losses.keys())
                    for k in tracked_loss_keys:
                        data[k] = []
                    print(f"  -> Auto-detected loss keys: {tracked_loss_keys}")

                if not all(k in temp_losses for k in tracked_loss_keys):
                    continue

                data["step"].append(s)
                data["epoch"].append(e)
                for k in tracked_loss_keys:
                    data[k].append(temp_losses[k])

                valid_count += 1

            except (json.JSONDecodeError, ValueError):
                continue

    print(f"  -> Extracted {valid_count} valid lines from {file_path}")
    return data if data["step"] else None


def read_teacher_log(file_path):
    data = {
        "step": [],
        "epoch": [],
        "loss": [],
        "grad_norm": [],
        "learning_rate": [],
        "mean_token_accuracy": []
    }

    if not file_path.exists():
        print(f"[Warning] File not found: {file_path}")
        return None

    print(f"Reading {file_path}...")
    valid_count = 0
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)

                # 严格提取
                s = int(entry["step"])
                e = float(entry["epoch"])
                l = float(entry["loss"])
                gn = float(entry["grad_norm"])
                lr = float(entry["learning_rate"])
                acc = float(entry["mean_token_accuracy"])

                data["step"].append(s)
                data["epoch"].append(e)
                data["loss"].append(l)
                data["grad_norm"].append(gn)
                data["learning_rate"].append(lr)
                data["mean_token_accuracy"].append(acc)
                valid_count += 1

            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    print(f"  -> Extracted {valid_count} valid lines from {file_path}")
    return data if data["step"] else None


def downsample_data(data, max_points):
    if data is None: return None
    total_points = len(data["step"])
    if total_points <= max_points:
        return data

    stride = total_points // max_points
    indices = list(range(0, total_points, stride))
    if indices[-1] != total_points - 1:
        indices.append(total_points - 1)

    print(f"  -> Downsampling: {total_points} points to {len(indices)}")

    new_data = {}
    for key, val_list in data.items():
        new_data[key] = [val_list[i] for i in indices]
    return new_data


def plot_distill_curves(data):
    if data is None: return

    steps = data["step"]

    plt.figure(figsize=(10, 6))

    plotted_count = 0
    for key, values in data.items():
        if key.startswith("loss_"):
            label_name = key.replace("loss_", "").replace("_", " ").title()
            plt.plot(steps, values, label=label_name, linewidth=1.5, alpha=0.8)
            plotted_count += 1

    plt.xlabel("Step")
    plt.ylabel("Loss Value")
    plt.title("Distillation Losses")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()


def plot_teacher_curves(data):
    if data is None: return

    steps = data["step"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Teacher Training Metrics", fontsize=16)

    axes[0, 0].plot(steps, data["loss"], color='tab:red')
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, data["mean_token_accuracy"], color='tab:green')
    axes[0, 1].set_title("Mean Token Accuracy")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, data["grad_norm"], color='tab:orange')
    axes[1, 0].set_title("Gradient Norm")
    axes[1, 0].set_ylabel("Norm")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, data["learning_rate"], color='tab:blue')
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()


def main():
    distill_data = read_distill_log(DISTILL_FILE)
    if distill_data:
        distill_data = downsample_data(distill_data, MAX_POINTS)
        plot_distill_curves(distill_data)

    teacher_data = read_teacher_log(TEACHER_FILE)
    if teacher_data:
        teacher_data = downsample_data(teacher_data, MAX_POINTS)
        plot_teacher_curves(teacher_data)

    if distill_data or teacher_data:
        plt.show()
    else:
        print("No valid data found.")


if __name__ == "__main__":
    main()
