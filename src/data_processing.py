"""Data processing utilities for distill_logits training."""

import logging
import os
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

logger = logging.getLogger(__name__)


def load_and_preprocess_dataset(config):
    """Load and preprocess dataset from configuration."""
    if type(config["dataset"]["split"]) is str:
        dataset = load_dataset(config["dataset"]["name"],
                               config["dataset"]["subset"],
                               split=config["dataset"]["split"])
    elif type(config["dataset"]["split"]) is list:
        splits = []

        for split in config["dataset"]["split"]:
            splits.append(
                load_dataset(config["dataset"]["name"], config["dataset"]["subset"], split=split))

        dataset = concatenate_datasets(splits)

    dataset = dataset.shuffle(seed=config["dataset"]["seed"])
    if "num_samples" in config["dataset"]:
        dataset = dataset.select(range(config["dataset"]["num_samples"]))
    return dataset


def mbpp_format(example, tokenizer, config, mode="train"):
    if mode == "train":
        message = [
            {
                "role": "user",
                "content": example['text']
            },
            {
                "role": "assistant",
                "content": example["code"]
            },
        ]
        add_generation_prompt = False
    else:
        message = [{"role": "user", "content": example['text']}]
        add_generation_prompt = True

    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    # Return formatted text along with original fields for later use
    return {
        "text": text,
        "Question": example['text'],
        "Response": example['code'],
        "Test": example['test_list']
    }


def add_assistant_labels(example, tokenizer):
    input_ids = example["input_ids"]
    labels = [-100] * len(input_ids)

    # NOTE: only for Qwen model
    assistant_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>assistant")
    assistant_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    in_assistant = False

    for i, token_id in enumerate(input_ids):
        if token_id == assistant_start_id:
            in_assistant = True
            continue

        if token_id == assistant_end_id and in_assistant:
            in_assistant = False
            continue

        if in_assistant:
            labels[i] = token_id

    example["labels"] = labels
    return example


def tokenize_function(examples, student_tokenizer, config):
    return student_tokenizer(
        examples["text"],
        truncation=True,
        max_length=config["tokenizer"]["max_length"],
        padding=False,
    )


def get_dataset_cache_dir(config):
    """Get the local cache directory for processed datasets."""
    cache_dir = config.get("dataset_cache_dir", "./dataset_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_split_cache_path(config, split="train"):
    """Get the cache file path for a specific split."""
    cache_dir = get_dataset_cache_dir(config)
    dataset_name = config["dataset"]["name"].replace("/", "_")
    return os.path.join(cache_dir, f"{dataset_name}_{split}")


def load_dataset_split(config, student_tokenizer, split="train"):
    """Load dataset split from cache or process if not cached.
    
    Args:
        config: Configuration dictionary
        student_tokenizer: Tokenizer for the student model
        split: "train" or "test"
    
    Returns:
        The dataset split (either from cache or newly processed)
    """
    split_path = get_split_cache_path(config, split)

    # Check if split exists in cache
    if os.path.exists(split_path):
        logger.info(f"Loading {split} split from cache: {split_path}")
        from datasets import load_from_disk
        return load_from_disk(split_path)

    # If cache doesn't exist, process the full dataset
    logger.info(f"Cache not found for {split} split. Processing dataset...")
    dataset = load_and_preprocess_dataset(config)
    tokenized_dataset = prepare_dataset(dataset, student_tokenizer, config)

    # Return the requested split
    return tokenized_dataset[split]


def prepare_dataset(dataset, student_tokenizer, config, mode="train"):
    """Prepare dataset by formatting and tokenizing using apply_chat_template.
    
    Saves train and test splits locally for future use.
    Preserves original Question/Response fields for generation tasks.
    """
    logger.info("Formatting dataset with FreedomIntelligence format using apply_chat_template...")

    # Format dataset with apply_chat_template
    dataset = dataset.map(lambda x: mbpp_format(x, student_tokenizer, config, mode),
                          desc="Formatting mbpp dataset")
    logger.info("Dataset formatting complete")

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, student_tokenizer, config),
        batched=True,
        num_proc=8,
        remove_columns=["text"],
    )
    tokenized_dataset = tokenized_dataset.map(
        lambda x: add_assistant_labels(x, student_tokenizer),
        num_proc=8,
        desc="Adding assistant-only labels",
    )
    logger.info("Tokenization complete")

    # Split into train and test
    logger.info("Splitting dataset into train/test (90/10)...")
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1,
                                                           seed=config["dataset"]["seed"])
    logger.info(
        f"Dataset split complete: train={len(tokenized_dataset['train'])}, test={len(tokenized_dataset['test'])}"
    )

    # Save splits locally
    train_path = get_split_cache_path(config, "train")
    test_path = get_split_cache_path(config, "test")

    logger.info(f"Saving train split to {train_path}...")
    tokenized_dataset["train"].save_to_disk(train_path)
    logger.info(f"Saving test split to {test_path}...")
    tokenized_dataset["test"].save_to_disk(test_path)
    logger.info("Dataset splits saved to local cache")

    return tokenized_dataset
