import json
import logging

import os

from transformers import TrainerCallback


def setup_file_logging(logger: logging.Logger, output_dir: str, filename: str):
    """
    Attach a file handler to the root logger to save logs to a file.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, filename)

    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path):
            return

    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.info(f"✅ Log file successfully attached to: {log_path}")


class FileLoggerCallback(TrainerCallback):

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                log_entry = {"step": state.global_step, "epoch": state.epoch, **logs}
                f.write(json.dumps(log_entry) + "\n")
