import os
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
import yaml
from datetime import datetime

from .config import CONFIG
from .data_processing import load_dataset_split, sanity_check_dataset

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def freeze_student_spectrum(model, unfrozen_layers_file, logger):
    """Freeze student model layers based on spectrum configuration."""
    with open(unfrozen_layers_file, 'r') as file:
        unfrozen_layers = yaml.safe_load(file)['unfrozen_parameters']

    for name, param in model.named_parameters():
        if not any(name.startswith(layer) for layer in unfrozen_layers):
            param.requires_grad = False
        else:
            param.requires_grad = True

    logger.info(f"Froze layers based on spectrum configuration: {unfrozen_layers_file}")


class MultiLayerAdaptationLayer(torch.nn.Module):
    """Multi-layer adaptation layer for hidden state distillation.
    
    Projects student hidden states to teacher hidden state dimensions
    with layer-wise mapping between student and teacher layers.
    """

    def __init__(self,
                 student_dim,
                 teacher_dim,
                 num_student_layers,
                 num_teacher_layers,
                 dtype=torch.bfloat16):
        """Initialize adaptation layer with projection modules.
        
        Args:
            student_dim: Hidden dimension of student model
            teacher_dim: Hidden dimension of teacher model
            num_student_layers: Number of student transformer layers
            num_teacher_layers: Number of teacher transformer layers
            dtype: Data type for projections (default: bfloat16)
        """
        super().__init__()
        # Create linear projections for each student layer
        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(student_dim, teacher_dim, dtype=dtype)
            for _ in range(num_student_layers)
        ])
        # Create mapping from student layer indices to teacher layer indices
        self.layer_mapping = self.create_layer_mapping(num_student_layers, num_teacher_layers)
        self.dtype = dtype

    def create_layer_mapping(self, num_student_layers, num_teacher_layers):
        """Create proportional layer mapping between student and teacher models.
        
        Maps each student layer to the nearest teacher layer based on relative position.
        
        Args:
            num_student_layers: Number of student layers
            num_teacher_layers: Number of teacher layers
            
        Returns:
            Dictionary mapping student layer indices to teacher layer indices
        """
        return {
            i: round(i * (num_teacher_layers - 1) / (num_student_layers - 1))
            for i in range(num_student_layers)
        }

    def forward(self, student_hidden_states):
        """Project student hidden states to teacher dimensions.
        
        Args:
            student_hidden_states: List of hidden states from student model layers
            
        Returns:
            List of projected hidden states matching teacher dimensions
        """
        adapted_hidden_states = []
        for i, hidden_state in enumerate(student_hidden_states):
            if i >= len(self.projections):
                break
            # Project student hidden state to teacher dimension
            adapted_hidden_states.append(self.projections[i](hidden_state.to(self.dtype)))
        return adapted_hidden_states


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
                    import random
                    test_indices = random.sample(range(len(self.test_dataset)),
                                                 min(self.num_test_samples, len(self.test_dataset)))

                    total_loss = 0
                    for idx, sample_idx in enumerate(test_indices, 1):
                        sample = self.test_dataset[sample_idx]

                        # Prepare input
                        input_ids = torch.tensor([sample['input_ids']]).to(model.device)
                        attention_mask = torch.tensor([sample['attention_mask']]).to(model.device)

                        # Forward pass
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=torch.tensor([sample["labels"]]).to(model.device),
                        )

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
                                enable_thinking=False,
                            )
                            gen_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                            gen_output = model.generate(
                                **gen_inputs,
                                max_new_tokens=CONFIG["tokenizer"]["max_new_tokens"],
                                repetition_penalty=1.2,
                                no_repeat_ngram_size=6,
                            )
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


def pad_logits(student_logits, teacher_logits):
    """Pad logits to match dimensions."""
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size),
                                 dtype=teacher_logits.dtype,
                                 device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1),
                teacher_logits) if student_size < teacher_size else (
                    student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits


class LogitsTrainer(Trainer):
    """Custom trainer for combined logits and hidden state knowledge distillation."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute combined distillation loss from logits and hidden states.
        
        Args:
            model: Student model
            inputs: Input batch
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch
            
        Returns:
            Loss value or tuple of (loss, outputs) if return_outputs=True
        """
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        teacher_model = self.teacher_model.module if hasattr(self.teacher_model,
                                                             'module') else self.teacher_model

        # Get student outputs with hidden states for hidden state distillation
        student_outputs = model(**inputs, output_hidden_states=True)
        with torch.no_grad():
            # Get teacher outputs with hidden states
            teacher_outputs = teacher_model(**inputs, output_hidden_states=True)

        custom_loss = self.distillation_loss(model, student_outputs, teacher_outputs, inputs,
                                             student_outputs.loss)
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, model, student_outputs, teacher_outputs, inputs, original_loss):
        """Compute combined logits and hidden state distillation loss.
        
        Combines:
        1. Logits-based KL divergence loss
        2. Hidden state-based distillation loss (if adaptation layer is available)
        
        Args:
            model: Student model
            student_outputs: Student model outputs including logits and hidden states
            teacher_outputs: Teacher model outputs including logits and hidden states
            inputs: Input batch
            original_loss: Original task loss from student model
            
        Returns:
            Combined distillation loss
        """
        device = next(model.parameters()).device
        T = self.config_dict["distillation"]["temperature"]

        # region: 1. Compute logits distillation loss
        student_logits, teacher_logits = pad_logits(student_outputs.logits.to(device),
                                                    teacher_outputs.logits.to(device))

        student_logits_scaled = student_logits / T
        teacher_logits_scaled = teacher_logits / T

        labels = inputs["labels"]  # [B, T]

        # KL Divergence
        # NOTE: log_softmax for student, softmax for teacher
        log_probs_student = F.log_softmax(student_logits_scaled, dim=-1)
        probs_teacher = F.softmax(teacher_logits_scaled, dim=-1)

        kl = F.kl_div(log_probs_student, probs_teacher, reduction="none").sum(dim=-1)  # [B, T]

        mask = (labels != -100).float()  # [B, T]

        loss_logits = (kl * mask).sum() / mask.sum().clamp_min(1)
        loss_logits = loss_logits * (T * T)
        # endregion

        # region: 2. Compute hidden state distillation loss
        loss_hidden = 0
        if hasattr(self, 'adaptation_layer') and self.adaptation_layer is not None:
            loss_hidden = self._compute_hidden_state_loss(student_outputs, teacher_outputs)
        # endregion

        # region: 3. Combine logits loss with hidden state loss
        distillation_weight = self.config_dict["distillation"].get("distillation_weight", 1.0)
        hidden_weight = self.config_dict["distillation"].get("hidden_weight", 0.5)

        combined_kd_loss = distillation_weight * loss_logits + hidden_weight * loss_hidden
        # endregion

        loss_logits_scalar = loss_logits.detach().float().item()
        loss_hidden_scalar = (loss_hidden.detach().float().item()
                              if torch.is_tensor(loss_hidden) else 0.0)
        original_loss_scalar = original_loss.detach().float().item()

        # logging (TensorBoard-safe)
        if self.state.global_step % 50 == 0:
            self.log({
                "loss_kd_logits": loss_logits_scalar,
                "loss_kd_hidden": loss_hidden_scalar,
                "loss_ce": original_loss_scalar,
            })

        # periodic console logging
        if self.state.global_step > 0 and self.state.global_step % 200 == 0:
            logger.info(f"KD logits: {loss_logits_scalar:.4f}, "
                        f"KD hidden: {loss_hidden_scalar:.4f}, "
                        f"CE: {original_loss_scalar:.4f}")

        alpha = self.config_dict["distillation"]["alpha"]

        # Weighted sum: alpha * KD + (1-alpha) * CE
        total_loss = alpha * combined_kd_loss + (1 - alpha) * original_loss

        return total_loss

    def _compute_hidden_state_loss(self, student_outputs, teacher_outputs):
        """Compute hidden state distillation loss using adaptation layer.
        
        Projects student hidden states to teacher dimensions and computes
        KL divergence loss for each mapped layer pair.
        
        Args:
            student_outputs: Student model outputs with hidden states
            teacher_outputs: Teacher model outputs with hidden states
            
        Returns:
            Averaged hidden state distillation loss
        """
        student_hidden_states = student_outputs.hidden_states
        teacher_hidden_states = teacher_outputs.hidden_states

        # Move adaptation layer to correct device
        device = student_hidden_states[0].device

        # Project student hidden states to teacher dimensions
        adapted_student_hidden_states = self.adaptation_layer(student_hidden_states)

        total_loss = 0
        num_layers = 0

        # Compute mse loss for each student-teacher layer pair
        for student_idx, teacher_idx in self.adaptation_layer.layer_mapping.items():
            if student_idx >= len(adapted_student_hidden_states):
                break

            student_hidden = adapted_student_hidden_states[student_idx]
            teacher_hidden = teacher_hidden_states[teacher_idx].to(device)

            # Verify shape compatibility
            if student_hidden.shape != teacher_hidden.shape:
                raise ValueError(
                    f"Shape mismatch: student {student_hidden.shape} vs teacher {teacher_hidden.shape}"
                )

            # Compute MSE loss (default reduction="mean")
            loss = F.mse_loss(F.layer_norm(student_hidden, student_hidden.shape[-1:]),
                              F.layer_norm(teacher_hidden, teacher_hidden.shape[-1:]))

            total_loss += loss
            num_layers += 1

        # Average loss across all layer pairs and normalize by hidden dimension
        if num_layers > 0:
            avg_loss = total_loss / num_layers
            return avg_loss

        return torch.tensor(0.0, device=device)


def main():
    """Main training function."""
    logger.info("Starting distill_logits training...")

    # Use configuration
    config = CONFIG

    # Set up environment
    os.environ['WANDB_PROJECT'] = config["project_name"]
    logger.info(f"Project name: {config['project_name']}")

    # Load tokenizers
    logger.info(
        f"Loading tokenizers: teacher={config['models']['teacher']}, student={config['models']['student']}"
    )
    student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])

    # Apply chat template to student tokenizer
    student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

    # Load train and test datasets (from cache if available, otherwise process)
    logger.info("Loading train dataset...")
    train_dataset = load_dataset_split(config, student_tokenizer, split="train")
    logger.info(f"Train dataset loaded with {len(train_dataset)} samples")

    logger.info("Loading test dataset...")
    test_dataset = load_dataset_split(config, student_tokenizer, split="test")
    logger.info(f"Test dataset loaded with {len(test_dataset)} samples")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=student_tokenizer, mlm=False)

    # Load models with configurable flash attention
    logger.info("Loading models...")
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using flash attention 2")

    teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"],
                                                         device_map="auto",
                                                         **model_kwargs)
    student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"],
                                                         device_map="auto",
                                                         **model_kwargs)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    logger.info("Models loaded successfully")

    # Optionally freeze layers of the student model based on spectrum configuration
    if "spectrum" in config and "layers_to_unfreeze" in config["spectrum"]:
        freeze_student_spectrum(student_model, config["spectrum"]["layers_to_unfreeze"], logger)
    else:
        logger.info(
            "Spectrum configuration not found. All layers of the student model will be trainable.")

    # Create adaptation layer for hidden state distillation
    logger.info("Creating multi-layer adaptation layer for hidden state distillation...")
    adaptation_layer = MultiLayerAdaptationLayer(
        student_dim=student_model.config.hidden_size,
        teacher_dim=teacher_model.config.hidden_size,
        num_student_layers=student_model.config.num_hidden_layers,
        num_teacher_layers=teacher_model.config.num_hidden_layers,
        dtype=torch.bfloat16)
    adaptation_layer = adaptation_layer.to(next(student_model.parameters()).device)
    logger.info(f"Adaptation layer created with {len(adaptation_layer.projections)} projections")
    student_model.adaptation_layer = adaptation_layer

    # Sanity check
    sanity_check_dataset(train_dataset, student_tokenizer)

    # Training arguments
    logger.info("Setting up training arguments...")
    training_arguments = TrainingArguments(**config["training"])

    # Create the custom SFT Trainer
    logger.info("Creating trainer...")
    trainer = LogitsTrainer(
        model=student_model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_arguments,
        data_collator=data_collator,
    )

    # Store config in trainer for access in loss computation
    trainer.config_dict = config

    # Add the teacher model to the trainer
    trainer.teacher_model = teacher_model

    # Add the adaptation layer to the trainer for hidden state distillation
    trainer.adaptation_layer = adaptation_layer

    # Add periodic test evaluation callback
    eval_steps = config["training"].get("eval_steps", 500)
    test_callback = PeriodicTestCallback(test_dataset=test_dataset,
                                         tokenizer=student_tokenizer,
                                         eval_steps=eval_steps,
                                         num_test_samples=5)
    trainer.add_callback(test_callback)
    logger.info(f"Added periodic test callback (every {eval_steps} steps)")

    # Train the model
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    # Save the final model
    logger.info(f"Saving model to {config['training']['output_dir']}")
    trainer.save_model(config["training"]["output_dir"])

    # Save the adaptation layer for hidden state distillation
    adaptation_layer_path = os.path.join(config["training"]["output_dir"], "adaptation_layer.pth")
    logger.info(f"Saving adaptation layer to {adaptation_layer_path}")
    torch.save(adaptation_layer.state_dict(), adaptation_layer_path)

    logger.info("Training completed successfully!")

    # Push to HuggingFace Hub if configured
    if config.get("hub", {}).get("push_to_hub", False):
        hub_config = config.get("hub", {})
        repo_name = hub_config.get("repo_name")

        if not repo_name:
            logger.error("push_to_hub is True but repo_name is not specified in config!")
        else:
            logger.info(f"Pushing model to HuggingFace Hub (repo: {repo_name})...")
            try:
                trainer.push_to_hub(repo_name=repo_name)
                logger.info(f"Model successfully pushed to HuggingFace Hub: {repo_name}")
            except Exception as e:
                logger.error(f"Error pushing model to HuggingFace Hub: {e}", exc_info=True)


if __name__ == "__main__":
    main()
