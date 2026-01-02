import json
import logging
import os
import re
import ast
from typing import Any, List, Dict, Optional

import numpy as np
import torch
import transformers
import yaml
from datasets import Dataset
from codebleu import calc_codebleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from sacrebleu import BLEU, CHRF
from tqdm import tqdm

from distillkit.configuration import (
    DatasetConfiguration,
    DistillationRunConfig,
    EvaluationConfig,
    TeacherModelConfig,
)
from distillkit.data_processing import load_data
from distillkit.logging_utils import setup_file_logging

LOG = logging.getLogger(__name__)

# Initialize NLTK data (if needed)
try:
    import nltk
    nltk.download("punkt", quiet=True)
except Exception:
    pass


def clean_code_generation(text: str) -> str:
    """
    Cleans generated text to extract code blocks if present.
    Useful for removing markdown artifacts (e.g., ```python) before calculating 
    AST or execution-based metrics.
    """
    # Pattern to find markdown code blocks
    pattern = r"```(?:\w+)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def calculate_ppl(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 4,
    device: str = "cuda",
    max_new_tokens: int = 2048,
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

            # Handle padding
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            labels = input_ids.clone()
            # Mask padding tokens in labels so they don't contribute to loss
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            num_tokens = attention_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            del outputs, loss, input_ids, attention_mask, labels
            torch.cuda.empty_cache()

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    ppl = np.exp(avg_loss)
    return float(ppl)


def calculate_bleu(
    predictions: List[str],
    references: List[str],
) -> float:
    """Calculate BLEU score using sacrebleu."""
    if len(predictions) == 0 or len(references) == 0:
        return 0.0

    bleu = BLEU()
    score = bleu.corpus_score(predictions, [references])
    return score.score / 100.0


def calculate_chrf(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Calculate ChrF score (Character n-gram F-score).
    Effective for code generation evaluation as it handles dense syntax.
    """
    if len(predictions) == 0 or len(references) == 0:
        return 0.0

    chrf = CHRF()
    score = chrf.corpus_score(predictions, [references])
    return score.score / 100.0


def calculate_ast_validity(predictions: List[str], language: str = "python") -> float:
    """
    Calculates the percentage of predictions that are syntactically valid 
    (parseable into an AST).
    Currently only supports Python using the built-in `ast` module.
    """
    if not predictions:
        return 0.0

    if language.lower() != "python":
        LOG.warning("AST validity check is currently only supported for Python.")
        return 0.0

    valid_count = 0
    for pred in predictions:
        # We must clean markdown tags before parsing
        clean_pred = clean_code_generation(pred)
        try:
            ast.parse(clean_pred)
            valid_count += 1
        except SyntaxError:
            continue
        except Exception:
            # Handle other encoding errors
            continue

    return valid_count / len(predictions)


def calculate_codebleu_score(predictions: List[str],
                             references: List[str],
                             language: str = "python") -> Dict[str, float]:
    """
    Calculate CodeBLEU score.
    CodeBLEU = Weighted combination of N-gram match, Weighted N-gram match (keywords),
               AST match (syntactic structure), and Data-flow match (semantic variables).
    """

    if not predictions or not references:
        return {"codebleu": 0.0}

    # Ensure inputs are clean strings (stripping markdown) for better parsing
    clean_preds = [clean_code_generation(p) for p in predictions]
    clean_refs = [clean_code_generation(r) for r in references]

    try:
        # Note: The 'tokenizer' arg might be required by some versions of codebleu,
        # but the standard package often defaults well.
        result = calc_codebleu(references=clean_refs,
                               predictions=clean_preds,
                               lang=language,
                               weights=(0.25, 0.25, 0.25, 0.25),
                               tokenizer=None)
        return result
    except Exception as e:
        LOG.error(f"Error calculating CodeBLEU: {e}")
        return {"codebleu": 0.0}


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
    max_length: int = 2048,
    device: str = "cuda",
) -> tuple[List[str], List[str], List[str]]:
    """
    Generate texts and return predictions, references, and prompts.
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

        # Case 1: Structured "messages" (Chat)
        if "messages" in batch or "conversations" in batch:
            raw_data = batch.get("messages", batch.get("conversations"))
            for conversation in raw_data:
                if isinstance(conversation, list) and len(conversation) > 0:
                    context_msgs = conversation[:-1]
                    target_msg = conversation[-1]

                    try:
                        prompt_str = tokenizer.apply_chat_template(
                            context_msgs,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except Exception:
                        # Fallback manual template
                        prompt_str = ""
                        for msg in context_msgs:
                            prompt_str += f"<|im_start|>{msg.get('role')}\n{msg.get('content')}<|im_end|>\n"
                        prompt_str += "<|im_start|>assistant\n"

                    if not prompt_str.endswith("\n"):
                        prompt_str += "\n"
                    prompts.append(prompt_str)

                    batch_references.append(target_msg.get("content", ""))
                else:
                    prompts.append("")
                    batch_references.append("")

        # Case 2: Plain text (Completion)
        elif "text" in batch:
            texts = batch["text"] if isinstance(batch["text"], list) else [batch["text"]]
            for text in texts:
                encoded = tokenizer(text,
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=max_length)
                ids = encoded["input_ids"][0]
                if len(ids) > 10:
                    split_point = int(len(ids) * 0.8)
                    prompt_text = tokenizer.decode(ids[:split_point], skip_special_tokens=True)
                    ref_text = tokenizer.decode(ids[split_point:], skip_special_tokens=True)
                else:
                    prompt_text = text
                    ref_text = ""
                prompts.append(prompt_text)
                batch_references.append(ref_text)
        else:
            continue

        if not prompts:
            continue

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        batch_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        predictions.extend(batch_preds)
        references.extend(batch_references)
        all_prompts.extend(prompts)

    return predictions, references, all_prompts


def evaluate_model(  # noqa: C901
    model_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset: Dataset,
    eval_config: EvaluationConfig,
    model_name: str = "model",
    teacher_config: Optional[TeacherModelConfig] = None,
) -> Dict[str, Any]:
    """Evaluate a single model with comprehensive metrics."""
    LOG.info(f"Evaluating {model_name} at {model_path}")
    results = {"model_name": model_name, "model_path": model_path}

    # Load Model (supports 4bit if teacher config present)
    try:
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if eval_config.device == "cuda" else torch.float32,
            "device_map": eval_config.device,
        }
        if teacher_config and teacher_config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type=teacher_config.bnb_4bit_quant_type or "nf4")
            load_kwargs.pop("torch_dtype", None)

        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        LOG.error(f"Failed to load model {model_path}: {e}")
        return results

    # 1. PPL Calculation
    try:
        ppl = calculate_ppl(model, tokenizer, dataset, eval_config.batch_size, eval_config.device,
                            eval_config.max_new_tokens)
        results["ppl"] = ppl
        LOG.info(f"{model_name} PPL: {ppl:.4f}")
    except Exception as e:
        LOG.error(f"PPL Error: {e}")

    # 2. Generation & Metrics
    try:
        predictions, references, _ = generate_texts(model, tokenizer, dataset,
                                                    eval_config.batch_size,
                                                    eval_config.max_new_tokens, eval_config.device)

        valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]

        if valid_pairs:
            preds, refs = zip(*valid_pairs)
            preds, refs = list(preds), list(refs)

            # Standard NLP Metrics
            results["bleu"] = calculate_bleu(preds, refs)
            results["chrf"] = calculate_chrf(preds, refs)
            results["rouge"] = calculate_rouge(preds, refs)
            results["f1"] = calculate_f1(preds, refs)

            # Code-Specific Metrics
            # AST Validity (Syntax Error Rate)
            ast_validity = calculate_ast_validity(preds, language="python")
            results["ast_validity"] = ast_validity
            LOG.info(f"{model_name} AST Validity (Syntax Check): {ast_validity:.4f}")

            # CodeBLEU (Composite Metric)
            codebleu_res = calculate_codebleu_score(preds, refs, language="python")
            results["codebleu"] = codebleu_res.get("codebleu", 0.0)
            LOG.info(f"{model_name} CodeBLEU: {results['codebleu']:.4f}")
            # Log detailed CodeBLEU sub-scores
            if "ngram_match_score" in codebleu_res:
                LOG.info(f"  - N-gram Match: {codebleu_res['ngram_match_score']:.4f}")
                LOG.info(f"  - Weighted N-gram: {codebleu_res['weighted_ngram_match_score']:.4f}")
                LOG.info(f"  - Syntax Match (AST): {codebleu_res['syntax_match_score']:.4f}")
                LOG.info(f"  - Dataflow Match: {codebleu_res['dataflow_match_score']:.4f}")

    except Exception as e:
        LOG.error(f"Metric calculation failed: {e}")

    del model
    torch.cuda.empty_cache()
    return results


def evaluate_all_models(
    config: EvaluationConfig,
    dataset_config: DatasetConfiguration,
    teacher_config: TeacherModelConfig,
) -> Dict[str, Any]:
    """Evaluate all models defined in the configuration."""
    os.makedirs(config.output_path, exist_ok=True)
    setup_file_logging(LOG, config.output_path, "evaluation.log")

    tokenizer_path = config.tokenizer_path

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset, _ = load_data(dataset_config, tokenizer)
    if config.num_samples:
        dataset = dataset.select(range(min(config.num_samples, len(dataset))))

    all_results = {}
    model_configs = [
        ("original_teacher", config.original_teacher_path, teacher_config),
        ("trained_teacher", config.trained_teacher_path, teacher_config),
        ("original_student", config.original_student_path, None),
        ("distilled_student", config.distilled_student_path, None),
    ]

    for name, path, t_conf in model_configs:
        if path:
            all_results[name] = evaluate_model(path, tokenizer, dataset, config, name, t_conf)

    # Save results
    with open(os.path.join(config.output_path, "evaluation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


def do_evaluate(config_path: str):  # noqa: C901
    """Main evaluation entry point."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Parse Configs (Simplified for brevity)
    eval_dict = config_dict.get("evaluation", config_dict)
    eval_config = EvaluationConfig(
        output_path=eval_dict.get("output_path", "outputs/evaluation_results"),
        max_new_tokens=eval_dict.get("max_new_tokens", 2048),
        batch_size=eval_dict.get("batch_size", 4),
        num_samples=eval_dict.get("num_samples"),
        device=eval_dict.get("device", "cuda"),
        tokenizer_path=eval_dict.get("tokenizer_path"),
        original_teacher_path=eval_dict.get("original_teacher_path"),
        trained_teacher_path=eval_dict.get("trained_teacher_path"),
        original_student_path=eval_dict.get("original_student_path"),
        distilled_student_path=eval_dict.get("distilled_student_path"),
    )

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

    # Extract teacher config for 4bit quantization
    if isinstance(distill_config.teacher, TeacherModelConfig):
        teacher_config = distill_config.teacher

    dataset_config = DatasetConfiguration.model_validate(config_dict.get("dataset", {}))

    teacher_config = TeacherModelConfig.model_validate(config_dict.get("teacher", {}))

    # Run Evaluation
    results = evaluate_all_models(eval_config, dataset_config, teacher_config)

    # Print Summary
    LOG.info("=" * 60)
    LOG.info("Final Evaluation Summary")
    LOG.info("=" * 60)
    for name, res in results.items():
        LOG.info(f"\nModel: {name}")
        if "ppl" in res: LOG.info(f"  PPL: {res['ppl']:.4f}")
        if "bleu" in res: LOG.info(f"  BLEU: {res['bleu']:.4f}")
        if "chrf" in res: LOG.info(f"  ChrF: {res['chrf']:.4f}")
        if "ast_validity" in res: LOG.info(f"  AST Validity: {res['ast_validity']:.4f}")
        if "codebleu" in res: LOG.info(f"  CodeBLEU: {res['codebleu']:.4f}")
    LOG.info("=" * 60)


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
