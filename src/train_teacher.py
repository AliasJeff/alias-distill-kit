"""Training script for the teacher model using SFT (Supervised Fine-Tuning)."""

import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from datetime import datetime
import random

from .config import CONFIG
from .data_processing import load_dataset_split, sanity_check_dataset

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PeriodicTestCallback(TrainerCallback):
    """Callback to run periodic test evaluation during training."""

    def __init__(self, test_dataset, tokenizer, eval_steps=500, num_test_samples=5):
        """Initialize the callback.
        
        Args:
            test_dataset: The test dataset to evaluate on (includes Question/Response fields)
            tokenizer: The tokenizer to use for generation
            eval_steps: Number of steps between evaluations
            num_test_samples: Number of samples to test
        """
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.num_test_samples = num_test_samples
        self.test_results = []

    def on_step_end(self, args, state, control, **kwargs):  # noqa: C901
        """Called at the end of each training step."""
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            model = kwargs.get('model')
            tokenizer = kwargs.get('tokenizer')
            if model is None or tokenizer is None:
                return

            logger.info(f"\n{'='*70}")
            logger.info(f"PERIODIC TEST EVALUATION - Step {state.global_step}")
            logger.info(f"{'='*70}")

            try:
                model.eval()
                with torch.no_grad():
                    test_indices = random.sample(range(len(self.test_dataset)),
                                                 min(self.num_test_samples, len(self.test_dataset)))

                    total_loss = 0
                    for idx, sample_idx in enumerate(test_indices, 1):
                        sample = self.test_dataset[sample_idx]

                        # Prepare input
                        input_ids = torch.tensor([sample['input_ids']]).to(model.device)
                        attention_mask = torch.tensor([sample['attention_mask']]).to(model.device)

                        # Forward pass
                        labels = torch.tensor([sample["labels"]]).to(model.device)
                        outputs = model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        labels=labels)

                        if isinstance(outputs, dict):
                            loss = outputs.get("loss", None)
                        else:
                            loss = getattr(outputs, "loss", None)

                        if loss is not None:
                            total_loss += float(loss.detach().cpu().item())
                            logger.info(
                                f"  Sample {idx}/{self.num_test_samples}: Loss = {loss:.4f}")

                        decoded_input = tokenizer.decode(sample["input_ids"],
                                                         skip_special_tokens=True)

                        decoded_target = None
                        if "labels" in sample:
                            decoded_target = tokenizer.decode([
                                t if t != -100 else tokenizer.pad_token_id for t in sample["labels"]
                            ],
                                                              skip_special_tokens=True)

                        logger.info("\n------ Q&A Example ------")
                        logger.info(f"Prompt:\n{decoded_input}")

                        if decoded_target:
                            logger.info(f"\nTarget:\n{decoded_target}")

                        try:
                            messages = [{"role": "user", "content": sample["Question"]}]
                            prompt = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                            gen_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                            gen_output = model.generate(**gen_inputs,
                                                        max_new_tokens=CONFIG["tokenizer"]
                                                        ["max_new_tokens"])
                            decoded_gen = tokenizer.decode(gen_output[0], skip_special_tokens=True)
                            logger.info(f"\nModel Output:\n{decoded_gen}")
                        except Exception as ge:
                            logger.warning(f"Generation error: {ge}")

                    # Average loss
                    avg_loss = total_loss / self.num_test_samples if self.num_test_samples > 0 else 0
                    logger.info(f"\nAverage Test Loss: {avg_loss:.4f}")

                    # Save result
                    self.test_results.append({
                        "step": state.global_step,
                        "avg_loss": avg_loss,
                        "timestamp": datetime.now().isoformat()
                    })

                model.train()

            except Exception as e:
                logger.error(f"Error during periodic test evaluation: {e}", exc_info=True)
                model.train()

            logger.info(f"{'='*70}\n")


def main():
    """Main training function for teacher model."""
    logger.info("Starting teacher model training...")

    # Use configuration
    config = CONFIG

    # Set up environment
    os.environ['WANDB_PROJECT'] = f"{config['project_name']}-teacher"
    logger.info(f"Project name: {config['project_name']}-teacher")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config['models']['teacher_origin']}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher_origin"])

    # Apply chat template to tokenizer
    teacher_tokenizer.chat_template = config["tokenizer"]["chat_template"]

    # Load train and test datasets (from cache if available, otherwise process)
    logger.info("Loading train dataset...")
    train_dataset = load_dataset_split(config, teacher_tokenizer, split="train")
    logger.info(f"Train dataset loaded with {len(train_dataset)} samples")

    logger.info("Loading test dataset...")
    test_dataset = load_dataset_split(config, teacher_tokenizer, split="test")
    logger.info(f"Test dataset loaded with {len(test_dataset)} samples")

    # Load teacher model
    logger.info("Loading teacher model...")
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using flash attention 2")

    teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher_origin"],
                                                         device_map="auto",
                                                         **model_kwargs)
    logger.info("Teacher model loaded successfully")

    # Create training configuration for teacher
    teacher_training_config = config["training"].copy()
    teacher_training_config["output_dir"] = teacher_training_config["output_dir"].replace(
        "results", "results_teacher")

    # Sanity check
    sanity_check_dataset(train_dataset, teacher_tokenizer)

    # Training arguments
    logger.info("Setting up training arguments...")
    training_arguments = TrainingArguments(**teacher_training_config)

    # Create the SFT Trainer for teacher model
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=teacher_model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_arguments,
        processing_class=teacher_tokenizer,
    )

    # Add periodic test evaluation callback
    eval_steps = config["training"].get("eval_steps", 500)
    test_callback = PeriodicTestCallback(test_dataset=test_dataset,
                                         tokenizer=teacher_tokenizer,
                                         eval_steps=eval_steps,
                                         num_test_samples=5)
    trainer.add_callback(test_callback)
    logger.info(f"Added periodic test callback (every {eval_steps} steps)")

    # Train the model
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=teacher_training_config.get("resume_from_checkpoint"))

    # Save the final model
    output_dir = teacher_training_config["output_dir"]
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    logger.info("Teacher model training completed successfully!")

    # Push to HuggingFace Hub if configured
    if config.get("hub", {}).get("push_to_hub", False):
        hub_config = config.get("hub", {})
        repo_name = hub_config.get("repo_name_teacher")

        if not repo_name:
            logger.error("push_to_hub is True but repo_name_teacher is not specified in config!")
        else:
            logger.info(f"Pushing teacher model to HuggingFace Hub (repo: {repo_name})...")
            try:
                trainer.push_to_hub(repo_name=repo_name)
                logger.info(f"Teacher model successfully pushed to HuggingFace Hub: {repo_name}")
            except Exception as e:
                logger.error(f"Error pushing teacher model to HuggingFace Hub: {e}", exc_info=True)


if __name__ == "__main__":
    main()
