import json
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DISTILL_FILE = Path("distill.jsonl")
TEACHER_FILE = Path("teacher_training.jsonl")
EVAL_FILE = Path("evaluation_results.json")
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
                if "step" not in entry or "epoch" not in entry: continue

                s = int(entry["step"])
                e = float(entry["epoch"])
                temp_losses = {}

                for key, value in entry.items():
                    match = pattern.search(key)
                    if match:
                        loss_name = match.group(2)
                        data_key = f"loss_{loss_name}"
                        temp_losses[data_key] = float(value)

                if not temp_losses: continue

                if tracked_loss_keys is None:
                    tracked_loss_keys = list(temp_losses.keys())
                    for k in tracked_loss_keys:
                        data[k] = []
                    print(f"  -> Auto-detected loss keys: {tracked_loss_keys}")

                if not all(k in temp_losses for k in tracked_loss_keys): continue

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

    lengths = [len(v) for v in data.values()]
    if len(set(lengths)) > 1:
        print(f"  [Error] Data length mismatch detected! Lengths: {lengths}")
        min_len = min(lengths)
        for k in data:
            data[k] = data[k][:min_len]

    return data if data["step"] else None


def read_evaluation_results(file_path):
    if not file_path.exists():
        print(f"[Warning] File not found: {file_path}")
        return []

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for key, val in data.items():
        item = {
            "name": val["model_name"],
            "ppl": val["ppl"],
            "bleu": val["bleu"],
            "f1": val["f1"],
            "rouge1": val["rouge"]["rouge1"]
        }
        results.append(item)

    print(f"  -> Loaded evaluation results for {len(results)} models.")
    return results


def downsample_data(data, max_points):
    """
    Downsamples data to a fixed number of points using linear spacing.
    Ensures exactly 'max_points' are returned, preserving start and end.
    """
    if data is None: return None

    total_points = len(data["step"])

    # If we have fewer points than the limit, no need to downsample
    if total_points <= max_points:
        return data

    print(f"  -> Downsampling: {total_points} points to {max_points}")

    # Generate evenly spaced integer indices
    # linspace generates floats, so we cast to int (forcing unique indices)
    indices = np.linspace(0, total_points - 1, max_points).astype(int)

    # Ensure indices are unique (just in case total_points is close to max_points)
    indices = np.unique(indices)

    new_data = {}
    for key, val_list in data.items():
        # Use numpy array indexing for speed if possible, otherwise list comp
        new_data[key] = [val_list[i] for i in indices]

    return new_data


def plot_distill_curves(data):
    if data is None: return
    steps = data["step"]
    plt.figure(figsize=(10, 6))
    for key, values in data.items():
        if key.startswith("loss_"):
            label_name = key.replace("loss_", "").replace("_", " ").title()
            plt.plot(steps, values, label=label_name, linewidth=1.5, alpha=0.8)

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

    metrics = [("loss", "Training Loss", 'tab:red'),
               ("mean_token_accuracy", "Mean Token Accuracy", 'tab:green'),
               ("grad_norm", "Gradient Norm", 'tab:orange'),
               ("learning_rate", "Learning Rate", 'tab:blue')]

    for ax, (key, title, color) in zip(axes.flat, metrics):
        ax.plot(steps, data[key], color=color)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()


def split_models_by_role(models_data):
    teachers = []
    students = []

    for m in models_data:
        if "teacher" in m["name"].lower():
            teachers.append(m)
        else:
            students.append(m)

    return teachers, students


def plot_grouped_metrics(models_list, title_suffix):
    if not models_list:
        print(f"[Info] No models found for {title_suffix}, skipping plot.")
        return

    metric_keys = ['ppl', 'bleu', 'f1', 'rouge1']
    metric_labels = ['PPL', 'BLEU', 'F1', 'ROUGE-1']

    n_metrics = len(metric_keys)
    n_models = len(models_list)

    total_width = 0.8
    bar_width = total_width / n_models

    plt.figure(figsize=(10, 6))

    x_base = np.arange(n_metrics)

    for i, model in enumerate(models_list):
        values = [model[k] for k in metric_keys]

        x_positions = x_base + (i - (n_models - 1) / 2) * bar_width

        plt.bar(x_positions,
                values,
                width=bar_width,
                label=model['name'],
                alpha=0.85,
                edgecolor='white')

        for x, v in zip(x_positions, values):
            plt.text(x, v + 0.01 * v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title(f'Evaluation Metrics Comparison - {title_suffix}')
    plt.xticks(x_base, metric_labels)
    plt.legend(title="Models")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()


def main():
    # 1. Distill Logs
    distill_data = read_distill_log(DISTILL_FILE)
    if distill_data:
        distill_data = downsample_data(distill_data, MAX_POINTS)
        plot_distill_curves(distill_data)

    # 2. Teacher Logs
    teacher_data = read_teacher_log(TEACHER_FILE)
    if teacher_data:
        teacher_data = downsample_data(teacher_data, MAX_POINTS)
        plot_teacher_curves(teacher_data)

    # 3. Evaluation Results (Bar Charts)
    all_models = read_evaluation_results(EVAL_FILE)
    if all_models:
        teachers, students = split_models_by_role(all_models)

        plot_grouped_metrics(teachers, "Teachers")
        plot_grouped_metrics(students, "Students")

    if distill_data or teacher_data or all_models:
        plt.show()
    else:
        print("No valid data found to plot.")


if __name__ == "__main__":
    main()
