# Copyright 2025 Arcee AI
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
from distillkit.data_processing import load_data
from distillkit.hsd_mapping import HiddenStateMapping
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

    resume_from_checkpoint = config.training_args.get("resume_from_checkpoint", None)

    LOG.info("Starting training.")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint, )
    LOG.info(f"Finished training. Saving model to {config.output_path}.")
    trainer.save_model(config.output_path)
    LOG.info("Done.")


@click.command("distillkit-offline")
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
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = DistillationRunConfig.model_validate(config_dict)
    do_distill(config)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    main()
