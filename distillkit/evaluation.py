import gc
import json
import logging
import os
from typing import Any, List, Dict

import numpy as np
import torch
import transformers
import yaml
from datasets import Dataset
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from sacrebleu import BLEU
from tqdm import tqdm

from distillkit.configuration import (
    DatasetConfiguration,
    DistillationRunConfig,
    EvaluationConfig,
    TeacherModelConfig,
)
from distillkit.data_processing import load_data, sanity_check_dataset
from distillkit.logging_utils import setup_file_logging

LOG = logging.getLogger(__name__)

# Initialize NLTK data (if needed)
try:
    import nltk
    nltk.download("punkt", quiet=True)
except Exception:
    pass


def calculate_ppl(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 8,
    device: str = "cuda",
    max_new_tokens: int = 32768,
) -> float:
    """Calculate the Perplexity (PPL) of the model."""
    model.eval()
    model = model.to(device)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Calculating PPL"):
            batch = dataset[i:i + batch_size]

            # Process input
            if "input_ids" in batch:
                input_ids = torch.tensor(batch["input_ids"]).to(device)
            elif "text" in batch:
                texts = batch["text"] if isinstance(batch["text"], list) else [batch["text"]]
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    truncation=True,
                    max_length=max_new_tokens,
                )
                input_ids = encoded["input_ids"].to(device)
            else:
                continue

            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            # Ensure pad_token_id is set
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            labels = input_ids.clone()
            # Mask padding tokens in labels so they don't contribute to loss
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            # Calculate the number of valid tokens (excluding padding)
            num_tokens = attention_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    ppl = np.exp(avg_loss)
    return float(ppl)


def calculate_bleu(
    predictions: List[str],
    references: List[str],
) -> float:
    """Calculate BLEU score."""
    if len(predictions) == 0 or len(references) == 0:
        return 0.0

    bleu = BLEU()
    # Transpose references for sacrebleu if necessary (it expects list of references, where each item is a list of all refs for that sample)
    # However, corpus_score expects: corpus_score(sys, [ref1, ref2, ...]) where ref1 is a list of lines.
    # The simple input here implies 1 reference per prediction.
    score = bleu.corpus_score(predictions, [references])
    return score.score / 100.0  # Convert to 0-1 range


def calculate_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Calculate ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": float(np.mean(rouge1_scores)),
        "rouge2": float(np.mean(rouge2_scores)),
        "rougeL": float(np.mean(rougeL_scores)),
    }


def calculate_f1(
    predictions: List[str],
    references: List[str],
) -> float:
    """Calculate F1 score (based on token-level exact match)."""
    if len(predictions) == 0 or len(references) == 0:
        return 0.0

    total_f1 = 0.0
    valid_count = 0

    for pred, ref in zip(predictions, references):
        if not pred.strip() or not ref.strip():
            continue

        try:
            pred_tokens = set(word_tokenize(pred.lower()))
            ref_tokens = set(word_tokenize(ref.lower()))
        except Exception:
            # Fallback to simple split if tokenize fails
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            continue

        intersection = pred_tokens & ref_tokens

        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(intersection) / len(ref_tokens) if ref_tokens else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        total_f1 += f1
        valid_count += 1

    if valid_count == 0:
        return 0.0

    return float(total_f1 / valid_count)


def generate_texts(  # noqa: C901
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 8,
    max_length: int = 32768,
    device: str = "cuda",
) -> tuple[List[str], List[str], List[str]]:
    """
    Generate texts and return predictions, references, and prompts.
    Optimized to use apply_chat_template if structured data is available.
    """
    model.eval()
    model = model.to(device)

    predictions = []
    references = []
    all_prompts = []

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating texts"):
        batch = dataset[i:i + batch_size]

        prompts = []
        batch_references = []

        # Case 1: Structured "messages" or "conversations" (Ideal for Chat Models)
        if "messages" in batch or "conversations" in batch:
            raw_data = batch.get("messages", batch.get("conversations"))

            for conversation in raw_data:
                if isinstance(conversation, list):
                    if len(conversation) > 0:
                        context_msgs = conversation[:-1]
                        target_msg = conversation[-1]

                        prompt_str = tokenizer.apply_chat_template(
                            context_msgs,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        if not prompt_str.endswith("\n"):
                            prompt_str += "\n"
                        prompts.append(prompt_str)

                        ref_content = target_msg.get("content", target_msg.get("value", ""))
                        batch_references.append(ref_content)
                    else:
                        prompts.append("")
                        batch_references.append("")
                else:
                    prompts.append("")
                    batch_references.append("")

        # Case 2: Pre-formatted "text" column (Fallback / Completion style)
        elif "text" in batch:
            texts = batch["text"] if isinstance(batch["text"], list) else [batch["text"]]

            # Here we must split the text blindly as we don't have the structure.
            # We use an 80/20 split as a heuristic for completion evaluation.
            for text in texts:
                encoded = tokenizer(text,
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=max_length)
                ids = encoded["input_ids"][0]

                if len(ids) > 10:
                    split_point = int(len(ids) * 0.8)
                    prompt_ids = ids[:split_point]
                    ref_ids = ids[split_point:]

                    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
                    ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
                else:
                    # Too short, just skip or use empty
                    prompt_text = text
                    ref_text = ""

                prompts.append(prompt_text)
                batch_references.append(ref_text)

        else:
            continue

        # Skip empty batches
        if not prompts:
            continue

        # Tokenize prompts
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=max_length,
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_length,
                num_beams=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[
                    tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id
                ],
            )

        # Decode generated text
        # We slice outputs to exclude the input prompt tokens
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]

        batch_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        predictions.extend(batch_preds)
        references.extend(batch_references)
        all_prompts.extend(prompts)

    return predictions, references, all_prompts


def evaluate_model(
    model_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset: Dataset,
    eval_config: EvaluationConfig,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Evaluate a single model."""
    LOG.info(f"Evaluating {model_name} at {model_path}")

    results = {"model_name": model_name, "model_path": model_path}

    # Load Model
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if eval_config.device == "cuda" else torch.float32,
            device_map=eval_config.device,  # Auto map to device
        )
    except Exception as e:
        LOG.error(f"Failed to load model {model_path}: {e}")
        return results

    # Calculate PPL
    try:
        ppl = calculate_ppl(
            model,
            tokenizer,
            dataset,
            batch_size=eval_config.batch_size,
            device=eval_config.device,
            max_new_tokens=eval_config.max_new_tokens,
        )
        results["ppl"] = ppl
        LOG.info(f"{model_name} PPL: {ppl:.4f}")
    except Exception as e:
        LOG.error(f"Failed to calculate PPL for {model_name}: {e}")
        results["ppl"] = None

    # Calculate Generation Metrics (BLEU, F1, ROUGE)
    try:
        predictions, references, _ = generate_texts(
            model,
            tokenizer,
            dataset,
            batch_size=eval_config.batch_size,
            max_length=eval_config.max_new_tokens,
            device=eval_config.device,
        )

        if len(predictions) > 0 and len(references) > 0:
            # Filter empty strings
            valid_pairs = [(p, r) for p, r in zip(predictions, references)
                           if p.strip() and r.strip()]

            if valid_pairs:
                valid_preds, valid_refs = zip(*valid_pairs)
                valid_preds = list(valid_preds)
                valid_refs = list(valid_refs)

                bleu = calculate_bleu(valid_preds, valid_refs)
                results["bleu"] = bleu
                LOG.info(f"{model_name} BLEU: {bleu:.4f}")

                f1 = calculate_f1(valid_preds, valid_refs)
                results["f1"] = f1
                LOG.info(f"{model_name} F1: {f1:.4f}")

                rouge = calculate_rouge(valid_preds, valid_refs)
                results["rouge"] = rouge
                LOG.info(f"{model_name} ROUGE: {rouge}")
            else:
                results.update({"bleu": None, "f1": None, "rouge": None})
        else:
            results.update({"bleu": None, "f1": None, "rouge": None})

    except Exception as e:
        LOG.error(f"Failed to calculate generation metrics for {model_name}: {e}")
        results.update({"bleu": None, "f1": None, "rouge": None})

    # Clean up memory
    del model
    if eval_config.device == "cuda":
        torch.cuda.empty_cache()

    return results


def evaluate_all_models(
    config: EvaluationConfig,
    dataset_config: DatasetConfiguration,
) -> Dict[str, Any]:
    """Evaluate all models defined in the configuration."""
    os.makedirs(config.output_path, exist_ok=True)

    setup_file_logging(LOG, config.output_path, "evaluation.log")

    # Load tokenizer
    # Use the first available model path to load the tokenizer
    tokenizer_path = (config.original_teacher_path or config.trained_teacher_path
                      or config.original_student_path or config.distilled_student_path)

    if tokenizer_path is None:
        raise ValueError("At least one model path must be provided")

    LOG.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Ensure chat template exists if we plan to use it, otherwise set a default
    # if tokenizer.chat_template is None:
    #     # Fallback to a simple ChatML-like template if none exists
    #     tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # Load dataset
    LOG.info("Loading dataset")
    # Note: load_data (from reference code) typically flattens data to 'text'.
    # If you want to use the 'messages' logic in generate_texts, ensure DatasetConfiguration
    # is set to NOT format everything into a single string immediately, or modify load_data.
    dataset, _ = load_data(dataset_config, tokenizer)

    if config.num_samples:
        dataset = dataset.select(range(min(config.num_samples, len(dataset))))

    all_results = {}

    model_configs = [
        ("original_teacher", config.original_teacher_path),
        ("trained_teacher", config.trained_teacher_path),
        ("original_student", config.original_student_path),
        ("distilled_student", config.distilled_student_path),
    ]

    for name, path in model_configs:
        if path:
            results = evaluate_model(
                path,
                tokenizer,
                dataset,
                config,
                name,
            )
            all_results[name] = results

    # Save results
    output_file = os.path.join(config.output_path, "evaluation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    LOG.info(f"Evaluation results saved to {output_file}")

    return all_results


def do_evaluate(config_path: str):  # noqa: C901
    """Main evaluation logic extracted from evaluate_main."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Try to read evaluation config from evaluation section, otherwise read from top level
    eval_dict = config_dict.get("evaluation", config_dict)

    # Try to parse as evaluation config
    try:
        eval_config = EvaluationConfig.model_validate(eval_dict)
    except Exception:
        # If parsing fails, create a minimal config
        eval_config = EvaluationConfig(
            output_path=eval_dict.get("output_path", "outputs/evaluation_results"),
            max_new_tokens=eval_dict.get("max_new_tokens"),
            batch_size=eval_dict.get("batch_size", 8),
            num_samples=eval_dict.get("num_samples"),
            device=eval_dict.get("device", "cuda"),
            original_teacher_path=eval_dict.get("original_teacher_path"),
            trained_teacher_path=eval_dict.get("trained_teacher_path"),
            original_student_path=eval_dict.get("original_student_path"),
            distilled_student_path=eval_dict.get("distilled_student_path"),
        )

    # If config contains distillation config, try to extract model paths from it
    try:
        distill_config = DistillationRunConfig.model_validate(config_dict)
        # If paths in evaluation config are empty, try to get them from distillation config
        if eval_config.original_teacher_path is None and isinstance(distill_config.teacher,
                                                                    TeacherModelConfig):
            eval_config.original_teacher_path = distill_config.teacher.path

        if eval_config.trained_teacher_path is None and distill_config.teacher_train:
            eval_config.trained_teacher_path = distill_config.teacher_train.output_path

        if eval_config.original_student_path is None:
            eval_config.original_student_path = distill_config.train_model

        if eval_config.distilled_student_path is None:
            eval_config.distilled_student_path = distill_config.output_path

        # Use dataset configuration from distillation config
        dataset_config = distill_config.dataset
    except Exception:
        # If parsing fails, try to get dataset config from evaluation config
        if "dataset" in config_dict:
            dataset_config = DatasetConfiguration.model_validate(config_dict["dataset"])
        elif "dataset" in eval_dict:
            dataset_config = DatasetConfiguration.model_validate(eval_dict["dataset"])
        else:
            raise ValueError("Dataset configuration is required in evaluation config")

    LOG.info("Performing pre-flight sanity check on evaluation data...")
    try:
        check_model_path = eval_config.distilled_student_path or eval_config.original_student_path

        if check_model_path:
            check_tokenizer = transformers.AutoTokenizer.from_pretrained(check_model_path,
                                                                         trust_remote_code=True)
            if check_tokenizer.pad_token_id is None:
                check_tokenizer.pad_token_id = check_tokenizer.eos_token_id

            check_ds, _ = load_data(dataset_config, check_tokenizer, keep_in_memory=True)

            sanity_check_dataset(check_ds, check_tokenizer)

            del check_tokenizer
            del check_ds
            gc.collect()
        else:
            LOG.warning("Skipping sanity check: No model path found in config to load tokenizer.")
    except Exception as e:
        LOG.warning(f"Sanity check failed (non-blocking): {e}")
        LOG.warning("Proceeding with evaluation anyway...")

    LOG.info("Starting evaluation")
    results = evaluate_all_models(eval_config, dataset_config)

    # Print summary
    LOG.info("\n" + "=" * 80)
    LOG.info("Evaluation Summary")
    LOG.info("=" * 80)
    for model_name, model_results in results.items():
        LOG.info(f"\n{model_name}:")
        if "ppl" in model_results and model_results["ppl"] is not None:
            LOG.info(f"  PPL: {model_results['ppl']:.4f}")
        if "bleu" in model_results and model_results["bleu"] is not None:
            LOG.info(f"  BLEU: {model_results['bleu']:.4f}")
        if "f1" in model_results and model_results["f1"] is not None:
            LOG.info(f"  F1: {model_results['f1']:.4f}")
        if "rouge" in model_results and model_results["rouge"] is not None:
            rouge = model_results["rouge"]
            LOG.info(f"  ROUGE-1: {rouge.get('rouge1', 0):.4f}")
            LOG.info(f"  ROUGE-2: {rouge.get('rouge2', 0):.4f}")
            LOG.info(f"  ROUGE-L: {rouge.get('rougeL', 0):.4f}")
    LOG.info("\n" + "=" * 80)
    LOG.info(f"Results saved to {eval_config.output_path}/evaluation_results.json")


def do_infer(  # noqa: C901
    config_path: str,
    model_path: str | None,
    num_samples: int | None,
    output_path: str | None,
    batch_size: int,
    max_new_tokens: int,
):
    """Main inference logic extracted from infer_main."""
    # Load config
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Try to parse as distillation config to get dataset and model info
    try:
        distill_config = DistillationRunConfig.model_validate(config_dict)
        dataset_config = distill_config.dataset
        if model_path is None:
            model_path = distill_config.train_model
        if output_path is None:
            output_path = "outputs/infer_results"
    except Exception:
        # If parsing fails, try to get dataset config separately
        if "dataset" in config_dict:
            dataset_config = DatasetConfiguration.model_validate(config_dict["dataset"])
        else:
            raise ValueError("Dataset configuration is required in config file")

        if model_path is None:
            if "model" in config_dict:
                model_path = config_dict["model"]
            elif "train_model" in config_dict:
                model_path = config_dict["train_model"]
            else:
                raise ValueError("Model path must be provided via --model-path or in config file")

    os.makedirs(output_path, exist_ok=True)
    setup_file_logging(LOG, output_path, "infer.log")

    LOG.info(f"Testing model: {model_path}")
    LOG.info(f"Output path: {output_path}")
    LOG.info(f"Number of samples: {num_samples or 'all'}")
    LOG.info(f"Batch size: {batch_size}")
    LOG.info(f"Max new tokens: {max_new_tokens}")

    # Load tokenizer
    LOG.info(f"Loading tokenizer from {model_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    LOG.info("Loading dataset")
    dataset, _ = load_data(dataset_config, tokenizer)

    # Limit samples if specified
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    elif dataset_config.num_samples:
        dataset = dataset.select(range(min(dataset_config.num_samples, len(dataset))))

    LOG.info(f"Dataset size: {len(dataset)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    LOG.info(f"Loading model from {model_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    model.eval()
    model = model.to(device)

    # Generate texts
    LOG.info("Starting text generation")
    with torch.no_grad():
        predictions, references, prompts = generate_texts(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            batch_size=batch_size,
            max_length=max_new_tokens,
            device=device,
        )

    # Save results
    results = []
    for i, (pred, ref, prompt) in enumerate(zip(predictions, references, prompts)):
        results.append({
            "sample_id": i,
            "prompt": prompt,
            "generated": pred,
            "reference": ref,
        })

    output_file = os.path.join(output_path, "infer_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Also save a human-readable text file
    text_output_file = os.path.join(output_path, "infer_results.txt")
    with open(text_output_file, "w", encoding="utf-8") as f:
        for i, result in enumerate(results):
            f.write(f"{'='*80}\n")
            f.write(f"Sample {i+1}\n")
            f.write(f"{'='*80}\n")
            if result.get("prompt"):
                f.write(f"Prompt:\n{result['prompt']}\n\n")
            f.write(f"Generated:\n{result['generated']}\n\n")
            if result.get("reference"):
                f.write(f"Reference:\n{result['reference']}\n\n")
            f.write("\n")

    LOG.info(f"Infer results saved to {output_file}")
    LOG.info(f"Human-readable results saved to {text_output_file}")
    LOG.info(f"Generated {len(predictions)} samples")

    # Clean up
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
