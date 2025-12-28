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
    TeacherDatasetConfig,
    TeacherModelConfig,
)
from distillkit.data_processing import load_data, sanity_check_dataset
from distillkit.evaluation import do_evaluate, do_infer
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.logging_utils import FileLoggerCallback, setup_file_logging
from distillkit.monkey_patch_packing import monkey_patch_packing_for_model
from distillkit.signals import OfflineSignalSource, OnlineSignalSource, SignalSource
from distillkit.trainer import DistillationTrainer

LOG = logging.getLogger(__name__)


class DataCollatorForCompletionOnlyLM(transformers.DataCollatorForLanguageModeling):

    def __init__(self, response_template, tokenizer, mlm=False):
        super().__init__(tokenizer, mlm=mlm)
        self.response_template = response_template
        self.tokenizer = tokenizer

        if isinstance(response_template, str):
            self.response_token_ids = self.tokenizer.encode(self.response_template,
                                                            add_special_tokens=False)
        else:
            self.response_token_ids = response_template

    def torch_call(self, examples):
        batch = super().torch_call(examples)

        labels = batch["labels"].clone()

        for i in range(len(examples)):
            response_token_ids_start_idx = None

            for idx in range(len(labels[i]) - len(self.response_token_ids) + 1):
                if torch.all(labels[i][idx:idx + len(self.response_token_ids)] == torch.tensor(
                        self.response_token_ids)):
                    response_token_ids_start_idx = idx
                    break

            if response_token_ids_start_idx is None:
                labels[i, :] = -100
            else:
                response_start = response_token_ids_start_idx + len(self.response_token_ids)
                labels[i, :response_start] = -100

        batch["labels"] = labels
        return batch


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

    response_template = "<|im_start|>assistant\n"
    response_template_ids = teacher_tokenizer.encode(response_template, add_special_tokens=False)

    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids,
                                               tokenizer=teacher_tokenizer)

    check_max_len = config.sequence_length
    sanity_check_dataset(
        ds_train,
        teacher_tokenizer,
        max_length=check_max_len,
        data_collator=collator,
    )

    teacher_trainer = trl.SFTTrainer(
        model=teacher_model,
        args=training_arguments,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=collate_packed_batch if teacher_dataset_config.prepacked else collator,
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

    response_template = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids,
                                               tokenizer=tokenizer)

    check_max_len = config.sequence_length
    sanity_check_dataset(
        ds_train,
        tokenizer,
        max_length=check_max_len,
        data_collator=collator,
    )

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
def evaluate_main(config_path: str, verbosity: int):
    log_level = logging.WARNING
    if verbosity >= 2:
        log_level = logging.DEBUG
    elif verbosity == 1:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)

    do_evaluate(config_path)


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
def infer_main(
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

    do_infer(
        config_path=config_path,
        model_path=model_path,
        num_samples=num_samples,
        output_path=output_path,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    main()
