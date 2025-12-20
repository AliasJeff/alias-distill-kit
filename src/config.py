"""Configuration for distillation training."""

CONFIG = {
    "project_name": "distillation",
    "dataset": {
        "name": "QuixiAI/leet10k-alpaca",
        "subset": None,
        "split": "train",  # input a single string or a list of split names
        "num_samples": 4000,  # You can pass a number here to limit the number of samples to use.
        "seed": 42,
        "format_function": "leet10k_format"
    },
    "models": {
        "teacher_origin": "Qwen/Qwen3-1.7B",
        "teacher": "results_teacher",
        "student": "Qwen/Qwen3-0.6B"
    },
    "tokenizer": {
        "max_length":
        38912,
        "max_new_tokens":
        32768,
        "chat_template": ("{%- for message in messages -%}"
                          "{%- if loop.first and messages[0]['role'] != 'system' -%}"
                          "<|im_start|>system\n"
                          "You are a helpful assistant. /no_think"
                          "<|im_end|>\n"
                          "{%- endif -%}"
                          "<|im_start|>{{ message['role'] }}\n"
                          "{{ message['content'] }}"
                          "<|im_end|>\n"
                          "{%- endfor -%}"
                          "{%- if add_generation_prompt -%}"
                          "<|im_start|>assistant\n"
                          "{%- endif -%}")
    },
    "training": {
        "output_dir": "results",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "save_steps": 1000,
        "eval_steps": 600,  # Evaluate every 600 steps
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint":
        None,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": True
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.5
    },
    "model_config": {
        "use_flash_attention": False
    },
    # "spectrum": {
    #     "layers_to_unfreeze": "/workspace/spectrum/snr_results_Qwen-Qwen2-1.5B_unfrozenparameters_50percent.yaml" # You can pass a spectrum yaml file here to freeze layers identified by spectrum.
    # }
    "gradio": {
        "port": 7860
    },
    "hub": {
        "push_to_hub": False,
        "repo_name": "qwen3-0.6b-medical-reasoning",
        "repo_name_teacher": "qwen3-1.7b-medical-reasoning"
    }
}
