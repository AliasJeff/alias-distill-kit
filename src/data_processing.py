"""Data processing utilities for distillation training."""

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
        enable_thinking=False,
    )

    # Return formatted text along with original fields for later use
    return {
        "text": text,
        "Question": example['text'],
        "Response": example['code'],
        "Test": example['test_list']
    }


def leet10k_format(example, tokenizer, config, mode="train"):
    if mode == "train":
        message = [
            {
                "role": "user",
                "content": f"{example['instruction']}\n{example['input']}"
            },
            {
                "role": "assistant",
                "content": example["output"]
            },
        ]
        add_generation_prompt = False
    else:
        message = [{"role": "user", "content": f"{example['instruction']}\n{example['input']}"}]
        add_generation_prompt = True

    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )

    # Return formatted text along with original fields for later use
    return {
        "text": text,
        "Question": f"{example['instruction']}\n{example['input']}",
        "Response": example['output']
    }


def add_assistant_labels(example, tokenizer):  # noqa: C901
    input_ids = example["input_ids"]
    labels = [-100] * len(input_ids)

    assistant_start = tokenizer("<|im_start|>assistant", add_special_tokens=False)["input_ids"]
    assistant_end = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id

    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(assistant_start)] == assistant_start:
            j = i + len(assistant_start)

            found_end = False
            while j < len(input_ids):
                if input_ids[j:j + len(assistant_end)] == assistant_end:
                    found_end = True

                    for k in range(j, j + len(assistant_end)):
                        labels[k] = input_ids[k]

                    search_start = j + len(assistant_end)
                    for k in range(search_start, len(input_ids)):
                        if input_ids[k] == eos_id:
                            labels[k] = eos_id
                            break

                    i = search_start
                    break

                labels[j] = input_ids[j]
                j += 1

            if not found_end:
                i = j
        else:
            i += 1

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
    func_name = config["dataset"]["format_function"]
    format_func = globals()[func_name]
    dataset = dataset.map(lambda x: format_func(x, student_tokenizer, config, mode),
                          desc="Formatting dataset")
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


def sanity_check_dataset(dataset, tokenizer, num_samples=3):  # noqa: C901
    logger.info("=" * 80)
    logger.info("RUNNING ENHANCED SANITY CHECK ON DATASET (EOS FOCUSED)")
    logger.info("=" * 80)

    assistant_start = tokenizer("<|im_start|>assistant", add_special_tokens=False)["input_ids"]
    assistant_end = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id

    for idx in range(num_samples):
        sample = dataset[idx]
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)

        logger.info(f"\n--- Sample {idx} ---")
        logger.info("FULL INPUT:")
        logger.info(decoded)

        # 1. Find assistant start and end
        start_pos = None
        end_pos = None

        for i in range(len(input_ids)):
            if input_ids[i:i + len(assistant_start)] == assistant_start:
                start_pos = i + len(assistant_start)
                break

        assert start_pos is not None, "❌ Assistant start not found"

        for j in range(start_pos, len(input_ids)):
            if input_ids[j:j + len(assistant_end)] == assistant_end:
                end_pos = j
                break

        assert end_pos is not None, "❌ Assistant end (<|im_end|>) not found"

        # 2. assistant content must be supervised
        supervised_content = [labels[k] != -100 for k in range(start_pos, end_pos)]
        assert any(supervised_content), "❌ No supervised assistant content"

        # 3. assistant_end must be supervised
        end_len = len(assistant_end)
        for k in range(end_pos, end_pos + end_len):
            assert labels[k] == input_ids[k], "❌ <|im_end|> is NOT supervised"

        logger.info("✅ <|im_end|> is supervised")

        # 4. EOS check (if exists)
        if eos_id is not None and eos_id in input_ids:
            eos_pos = len(input_ids) - 1 - input_ids[::-1].index(eos_id)  # Last EOS
            assert labels[eos_pos] == eos_id, "❌ EOS token exists but is NOT supervised"
            logger.info("✅ EOS token is supervised")

        # 5. Defensive check: user tokens should not be supervised
        user_start = tokenizer("<|im_start|>user", add_special_tokens=False)["input_ids"]
        for i in range(len(input_ids) - len(user_start)):
            if input_ids[i:i + len(user_start)] == user_start:
                assert all(labels[j] == -100 for j in range(i, i + len(user_start))), \
                    "❌ User tokens are supervised!"

        logger.info("✅ Sample passed EOS sanity check")

    logger.info("🎉 ALL SANITY CHECKS PASSED")
    logger.info("=" * 80)
