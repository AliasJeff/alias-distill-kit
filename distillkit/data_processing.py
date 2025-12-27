import hashlib
import json
import logging
import os
from typing import Any

import datasets
import transformers

from distillkit.configuration import (
    DatasetConfiguration,
    DatasetPath,
    HfRepoDataset,
    LocalDataset,
)

LOG = logging.getLogger(__name__)


def gpt_format(example, tokenizer):
    if "conversations" in example:
        conversations = example["conversations"]
        messages = []
        for conversation in conversations:
            role_map = {
                "human": "user",
                "user": "user",
                "gpt": "assistant",
                "assistant": "assistant",
                "system": "system",
            }
            role = role_map.get(conversation.get("from", ""), None)
            if role:
                messages.append({"role": role, "content": conversation.get("value", "")})

        # Apply chat template to create a single string.
        # SFTTrainer will handle tokenization.
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token

        return {"text": text, "messages": messages}
    else:
        raise RuntimeError("Expected `conversations` column")


def leet10k_format(example, tokenizer):
    output_content = example.get('output')

    messages = [
        {
            "role": "user",
            "content": f"{example['instruction']}\n\n{example['input']}"
        },
        {
            "role": "assistant",
            "content": output_content
        },
    ]

    if "Reference:" in output_content:
        print(f"!!! FOUND REFERENCE IN DATA !!!\nContent snippet: {output_content[:200]}...")
        output_content = output_content.split("Reference:")[0].strip()

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token

    return {"text": text, "messages": messages}


FORMAT_FUNCTIONS = {
    "gpt_format": gpt_format,
    "leet10k_format": leet10k_format,
}


def _format_row(
    example: dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizer,
    format_function: str | None = None,
) -> dict[str, Any]:
    if ("input_ids" in example) or ("text" in example):
        # either pretokenized or raw completion - no formatting needed
        return {}

    if format_function:
        fn = FORMAT_FUNCTIONS.get(format_function)
        if fn is None:
            raise RuntimeError(f"Unknown format_function: {format_function}")
        return fn(example, tokenizer)

    elif "messages" in example:
        text = tokenizer.apply_chat_template(example["messages"],
                                             tokenize=False,
                                             add_generation_prompt=False)
        if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
        return {"text": text, "messages": example["messages"]}
    else:
        raise RuntimeError("Expected `text`, `messages`, or `conversations` column")


def _load_dataset(  # noqa: C901
    path: DatasetPath,
    seed: int | None,
    num_samples: int | None,
    tokenizer: transformers.PreTrainedTokenizer,
    prepared_dataset_path: str | None = None,
    keep_in_memory: bool | None = None,
    prepacked: bool = False,
    format_function: str | None = None,
) -> datasets.Dataset:
    if prepared_dataset_path:
        honk = json.dumps({
            "path": path.model_dump(),
            "seed": seed,
            "num_samples": num_samples,
            "format_function": format_function,
        })
        logging.info(f"Dataset spec: {honk}")
        ds_hash = hashlib.sha256(honk.encode()).hexdigest()
        full_prepared_path = os.path.join(prepared_dataset_path, f"dataset-{ds_hash}")
        if os.path.exists(full_prepared_path):
            return datasets.load_from_disk(full_prepared_path)
    else:
        full_prepared_path = None
    if isinstance(path, HfRepoDataset):
        res = datasets.load_dataset(
            path.repo_id,
            name=path.config_name,
            revision=path.revision,
            split=path.split,
            keep_in_memory=keep_in_memory,
        )
    elif isinstance(path, LocalDataset):
        res = datasets.load_from_disk(path.disk_path, keep_in_memory=keep_in_memory)
        if path.split:
            res = res[path.split]
        elif isinstance(res, datasets.DatasetDict):
            raise ValueError("Dataset dict found but no split specified. Please specify a split.")
    else:
        raise ValueError(
            "Unsupported dataset type. Please provide a valid Hugging Face repo ID or local dataset path."
        )

    if prepacked:
        last_idx = len(res) - 1
        while len(res) >= 2 and len(res[last_idx]["input_ids"]) != len(res[0]["input_ids"]):
            last_idx -= 1
        if last_idx <= 0:
            raise RuntimeError("Dataset config is probs wrong")
        res = res.select(range(last_idx + 1))

    if seed:
        res = res.shuffle(seed=seed)
    if num_samples:
        res = res.select(range(num_samples))
    if ((not prepacked) and ("text" not in res.column_names)
            and ("input_ids" not in res.column_names)):
        res = res.map(
            _format_row,
            remove_columns=res.column_names,
            fn_kwargs={
                "tokenizer": tokenizer,
                "format_function": format_function,
            },
        )
    if full_prepared_path:
        os.makedirs(full_prepared_path, exist_ok=True)
        logging.info(
            f"Saving prepared dataset to {full_prepared_path} (hash: {ds_hash}, path: {path}, seed: {seed}, num_samples: {num_samples})"
        )
        res.save_to_disk(full_prepared_path)
        del res
        return datasets.load_from_disk(full_prepared_path, keep_in_memory=keep_in_memory)
    return res


def load_data(
    config: DatasetConfiguration,
    tokenizer: transformers.PreTrainedTokenizer,
    keep_in_memory: bool | None = None,
) -> tuple[datasets.Dataset, datasets.Dataset | None]:
    """
    Load the train (and optionally eval) datasets as specified in the configuration.
    """

    LOG.info(f"Loading datasets: {config.train_dataset} (train), {config.eval_dataset} (eval)")
    ds_train = _load_dataset(
        config.train_dataset,
        config.seed,
        config.num_samples,
        tokenizer=tokenizer,
        prepared_dataset_path=config.prepared_dataset_path,
        keep_in_memory=keep_in_memory,
        prepacked=config.prepacked,
        format_function=config.format_function,
    )
    ds_eval = None
    if config.eval_dataset:
        ds_eval = _load_dataset(
            config.eval_dataset,
            config.seed,
            config.num_eval_samples,
            tokenizer=tokenizer,
            prepared_dataset_path=config.prepared_dataset_path,
            keep_in_memory=keep_in_memory,
            prepacked=config.prepacked,
            format_function=config.format_function,
        )
    return ds_train, ds_eval
