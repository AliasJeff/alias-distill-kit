import json
import logging
import os
import re

import click
import torch
import transformers
import trl
import yaml
from accelerate import Accelerator

from distillkit.compression import LogprobCompressor
from distillkit.configuration import (
    DistillationRunConfig,
    EvaluationConfig,
    TeacherDatasetConfig,
    TeacherModelConfig,
)
from distillkit.data_processing import load_data
from distillkit.evaluation import evaluate_all_models, generate_texts
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.logging import FileLoggerCallback, setup_file_logging
from distillkit.monkey_patch_packing import monkey_patch_packing_for_model
from distillkit.signals import OfflineSignalSource, OnlineSignalSource, SignalSource
from distillkit.trainer import DistillationTrainer

LOG = logging.getLogger(__name__)


def load_student_model(  # noqa: C901
    config: DistillationRunConfig,
    tokenizer_vocab_size: int,
) -> transformers.PreTrainedModel:
    if config.functionary_packing:
        monkey_patch_packing_for_model(config.train_model)
    auto_cls = getattr(transformers, config.model_auto_class, None)
    if auto_cls is None:
        raise ValueError(f"Model class {config.model_auto_class} not found in transformers.")
    LOG.info(f"Loading model {config.train_model} with class {auto_cls}")
    extra_kwargs = {"trust_remote_code": config.trust_remote_code}
    if config.use_flash_attention:
        extra_kwargs["attn_implementation"] = "flash_attention_2"
        extra_kwargs["torch_dtype"] = torch.bfloat16
    model = auto_cls.from_pretrained(
        config.train_model,
        **extra_kwargs,
        **config.model_kwargs,
    )
    LOG.info("Loaded model.")

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    if (model_vocab_size != tokenizer_vocab_size or config.resize_embeddings_to_multiple_of):
        model.resize_token_embeddings(
            tokenizer_vocab_size,
            pad_to_multiple_of=config.resize_embeddings_to_multiple_of,
        )
        new_model_vocab_size = model.get_input_embeddings().weight.shape[0]
        if new_model_vocab_size != model_vocab_size:
            LOG.info(f"Resized model vocab size from {model_vocab_size} to {new_model_vocab_size}")

    model: transformers.PreTrainedModel
    if config.frozen_modules:
        module_set = set(config.frozen_modules)
        seen = set()
        for name, module in model.named_modules():
            if name in module_set:
                module.requires_grad_(False)
                seen.add(name)
        unseen = module_set - seen
        LOG.info(f"Froze {len(seen)} modules")
        if unseen:
            raise ValueError(f"Frozen modules not found in model: {', '.join(unseen)}")
    if config.frozen_res:
        num_frozen = 0
        frozen_res = [re.compile(s) for s in config.frozen_res]
        for name, param in model.named_parameters():
            if any(fre.search(name) for fre in frozen_res):
                param.requires_grad = False
                num_frozen += 1
        if num_frozen:
            print(f"Froze {num_frozen} tensors by regular expression")
    return model


def train_teacher(config: DistillationRunConfig, accelerator: Accelerator) -> None:
    if not isinstance(config.teacher, TeacherModelConfig):
        raise ValueError("Teacher must be a HF model (TeacherModelConfig) for training.")
    if config.teacher_train is None:
        raise ValueError("teacher_train configuration must be set to train the teacher model.")

    teacher_cfg = config.teacher_train

    os.makedirs(teacher_cfg.output_path, exist_ok=True)

    teacher_dataset_config = teacher_cfg.dataset or config.dataset

    with accelerator.main_process_first():
        teacher_tokenizer = load_tokenizer(config)
        ds_train, ds_eval = load_data(teacher_dataset_config, teacher_tokenizer)

    teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
        config.teacher.path,
        trust_remote_code=config.trust_remote_code,
        **(config.teacher.kwargs or {}),
    )

    teacher_training_args = dict(teacher_cfg.training_args)
    dataset_kwargs = teacher_training_args.pop("dataset_kwargs", {})
    if teacher_dataset_config.prepacked:
        dataset_kwargs["skip_prepare_dataset"] = True
    max_length = teacher_training_args.pop("max_length", config.sequence_length)
    training_arguments = trl.SFTConfig(
        **teacher_training_args,
        max_length=max_length,
        output_dir=teacher_cfg.output_path,
        dataset_kwargs=dataset_kwargs,
    )

    teacher_trainer = trl.SFTTrainer(
        model=teacher_model,
        args=training_arguments,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=collate_packed_batch if teacher_dataset_config.prepacked else None,
        processing_class=None if teacher_dataset_config.prepacked else teacher_tokenizer,
    )
    teacher_trainer.add_callback(
        FileLoggerCallback(os.path.join(teacher_cfg.output_path, "teacher_training.jsonl")))

    resume_from_checkpoint = teacher_cfg.training_args.get("resume_from_checkpoint", None)

    LOG.info("Starting teacher training.")
    teacher_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    LOG.info(f"Finished teacher training. Saving teacher model to {teacher_cfg.output_path}.")
    teacher_trainer.save_model(teacher_cfg.output_path)
    LOG.info("Done training teacher.")

    config.teacher.path = teacher_cfg.output_path


def create_signal_source(config: DistillationRunConfig, vocab_size: int) -> SignalSource:
    if isinstance(config.teacher, TeacherDatasetConfig):
        compressor = LogprobCompressor(
            config=config.teacher.logprob_compressor,
            legacy_config=config.teacher.legacy_logit_compression,
        )
        return OfflineSignalSource(compressor, vocab_size=vocab_size)
    elif isinstance(config.teacher, TeacherModelConfig):
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.teacher.path, **(config.teacher.kwargs or {}))
        return OnlineSignalSource(teacher_model,
                                  vocab_size=vocab_size,
                                  sparsify_top_k=config.teacher.top_k)
    else:
        raise RuntimeError("Teacher configuration invalid")


def collate_packed_batch(examples):
    # all sequences in the batch already have the same length
    # so we can directly stack them
    return {key: torch.tensor([example[key] for example in examples]) for key in examples[0].keys()}


def load_tokenizer(config: DistillationRunConfig) -> transformers.PreTrainedTokenizer:
    if isinstance(config.teacher, TeacherModelConfig):
        src_path = config.teacher.path
        logging.info("Using teacher's tokenizer")
    else:
        src_path = config.train_model
        logging.info("Using student's tokenizer")
    return transformers.AutoTokenizer.from_pretrained(
        src_path,
        trust_remote_code=config.trust_remote_code,
    )


def do_distill(config: DistillationRunConfig, config_source: str | None = None):
    os.makedirs(config.output_path, exist_ok=True)
    if config_source is None:
        config_source = yaml.safe_dump(config.model_dump(mode="json", by_alias=True))
    with open(os.path.join(config.output_path, "distillkit_config.yaml"), "w") as f:
        f.write(config_source)

    if config.project_name:
        os.environ["WANDB_PROJECT"] = config.project_name

    accelerator = Accelerator()
    if accelerator.is_main_process:
        setup_file_logging(LOG, config.output_path, "distill.log")

    with accelerator.main_process_first():
        tokenizer = load_tokenizer(config)
        ds_train, ds_eval = load_data(config.dataset, tokenizer)

        tokenizer_vocab_size = max(
            len(tokenizer.get_vocab()),
            max(tokenizer.get_vocab().values()) + 1,
        )

    model = load_student_model(config, tokenizer_vocab_size)

    config_kwargs = dict(config.training_args)
    dataset_kwargs = config_kwargs.pop("dataset_kwargs", {})
    if config.dataset.prepacked:
        dataset_kwargs["skip_prepare_dataset"] = True
    max_length = config_kwargs.pop("max_length", config.sequence_length)
    training_arguments = trl.SFTConfig(
        **config_kwargs,
        max_length=max_length,
        output_dir=config.output_path,
        dataset_kwargs=dataset_kwargs,
    )

    signal_source = create_signal_source(config, tokenizer_vocab_size)
    if config.layer_mapping is not None:
        if not isinstance(signal_source, OnlineSignalSource):
            raise RuntimeError("Hidden state distillation not supported for offline teachers")
        teacher_hidden_size = signal_source.teacher_model.config.hidden_size
        if config.layer_mapping == "all":
            mapping = [(i, i) for i in range(model.config.num_hidden_layers)]
        else:
            mapping = config.layer_mapping
        hsm = HiddenStateMapping(
            student=model,
            teacher_hidden_size=teacher_hidden_size,
            layer_mapping=mapping,
            force_projection=config.force_hidden_state_projection,
        )
    else:
        hsm = None
    trainer = DistillationTrainer(
        model=model,
        config=config,
        signal_source=signal_source,
        hidden_state_mapping=hsm,
        true_vocab_size=tokenizer_vocab_size,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        args=training_arguments,
        data_collator=collate_packed_batch if config.dataset.prepacked else None,
        processing_class=None if config.dataset.prepacked else tokenizer,
    )
    trainer.add_callback(FileLoggerCallback(os.path.join(config.output_path, "distill.jsonl")))

    resume_from_checkpoint = config.training_args.get("resume_from_checkpoint", None)

    LOG.info("Starting training.")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint, )
    LOG.info(f"Finished training. Saving model to {config.output_path}.")
    trainer.save_model(config.output_path)
    LOG.info("Done.")


@click.command("distill")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--verbose",
    "-v",
    "verbosity",
    count=True,
    help="Increase verbosity of logging. Use -vv for debug level.",
)
def main(config_path: str, verbosity: int):
    log_level = logging.WARNING
    if verbosity >= 2:
        log_level = logging.DEBUG
    elif verbosity == 1:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)

    # 1. Load Config
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = DistillationRunConfig.model_validate(config_dict)

    do_distill(config)


@click.command("train-teacher")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--verbose",
    "-v",
    "verbosity",
    count=True,
    help="Increase verbosity of logging. Use -vv for debug level.",
)
def train_teacher_main(config_path: str, verbosity: int):
    log_level = logging.WARNING
    if verbosity >= 2:
        log_level = logging.DEBUG
    elif verbosity == 1:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)

    # 1. Load Config
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = DistillationRunConfig.model_validate(config_dict)

    # 2. Setup File Logging for Teacher Training
    # Check if teacher_train config exists to avoid error, though validation handles it usually
    if config.teacher_train and config.teacher_train.output_path:
        setup_file_logging(LOG, config.teacher_train.output_path, "teacher_training.log")

    accelerator = Accelerator()
    train_teacher(config, accelerator)


@click.command("evaluate")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--verbose",
    "-v",
    "verbosity",
    count=True,
    help="Increase verbosity of logging. Use -vv for debug level.",
)
def evaluate_main(config_path: str, verbosity: int):  # noqa: C901
    log_level = logging.WARNING
    if verbosity >= 2:
        log_level = logging.DEBUG
    elif verbosity == 1:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)

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
            from distillkit.configuration import DatasetConfiguration
            dataset_config = DatasetConfiguration.model_validate(config_dict["dataset"])
        elif "dataset" in eval_dict:
            from distillkit.configuration import DatasetConfiguration
            dataset_config = DatasetConfiguration.model_validate(eval_dict["dataset"])
        else:
            raise ValueError("Dataset configuration is required in evaluation config")

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


@click.command("infer")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--model-path",
    type=str,
    help="Path to the model to test. If not provided, will try to extract from config.",
)
@click.option(
    "--num-samples",
    type=int,
    default=None,
    help="Number of samples to test. If not provided, uses all samples or config value.",
)
@click.option(
    "--output-path",
    type=str,
    default=None,
    help=
    "Path to save infer results. If not provided, uses config output_path or 'outputs/infer_results'.",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Batch size for generation.",
)
@click.option(
    "--max-new-tokens",
    type=int,
    default=32768,
    help="Maximum number of new tokens to generate.",
)
@click.option(
    "--verbose",
    "-v",
    "verbosity",
    count=True,
    help="Increase verbosity of logging. Use -vv for debug level.",
)
def infer_main(  # noqa: C901
    config_path: str,
    model_path: str | None,
    num_samples: int | None,
    output_path: str | None,
    batch_size: int,
    max_new_tokens: int,
    verbosity: int,
):
    """Infer model on dataset samples."""
    log_level = logging.WARNING
    if verbosity >= 2:
        log_level = logging.DEBUG
    elif verbosity == 1:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)

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
            from distillkit.configuration import DatasetConfiguration
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


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    main()
