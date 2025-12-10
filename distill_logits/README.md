# DistillLogits

Knowledge distillation framework for training efficient student models using logits-based distillation from teacher models.

## Overview

DistillLogits is a comprehensive knowledge distillation toolkit that enables you to:
- Train efficient student models by distilling knowledge from larger teacher models
- Use logits-based KL-divergence loss for effective knowledge transfer
- Evaluate model performance with multiple metrics (perplexity, BLEU, F1)
- Generate and test model outputs
- Compare original and distilled models

## Features

- **Logits-based Knowledge Distillation**: Uses KL-divergence loss on model logits with configurable temperature and alpha parameters
- **Flexible Model Support**: Works with any HuggingFace transformer model
- **Comprehensive Evaluation**: Computes perplexity, BLEU scores, and F1 scores
- **Model Comparison**: Compare original and distilled models side-by-side
- **Performance Testing**: Generate outputs and measure generation speed
- **Configurable Training**: Supports flash attention, mixed precision, gradient accumulation, and more

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA (recommended for GPU acceleration)

## Configuration

Edit `config.py` to customize your training setup.

### Key Configuration Options

- **Dataset**: Configure dataset name, language, and split
- **Models**: Set teacher and student model paths
- **Training**: Batch size, learning rate, epochs, evaluation frequency, etc.
- **Distillation**: Temperature and alpha parameters for KD loss
- **Hub**: Set `push_to_hub: true` to automatically upload trained models to HuggingFace Hub after training

#### HuggingFace Hub Upload

To enable automatic model upload to HuggingFace Hub after training:

```python
"hub": {
    "push_to_hub": True,
    "repo_name": "your-username/distil-logits-student",  # Student model repository
    "repo_name_teacher": "your-username/distil-logits-teacher"  # Teacher model repository
}
```

**Configuration Parameters:**
- `push_to_hub`: Set to `True` to enable automatic upload (default: `False`)
- `repo_name`: HuggingFace Hub repository name for the student model (e.g., `username/model-name`)
- `repo_name_teacher`: HuggingFace Hub repository name for the teacher model (e.g., `username/teacher-model-name`)

**Requirements:**
- HuggingFace CLI authentication: `huggingface-cli login`
- Repositories must exist on HuggingFace Hub before training
- Both `repo_name` and `repo_name_teacher` must be explicitly specified if `push_to_hub` is `True`
- Trainer will automatically push the model after training completes

**Example Configuration:**
```python
"hub": {
    "push_to_hub": True,
    "repo_name": "my-org/qwen-distilled-student",
    "repo_name_teacher": "my-org/qwen-distilled-teacher"
}
```

## Usage

### 0. Train the Teacher Model (Optional)

Train the teacher model first if you want to fine-tune it on your dataset:

```bash
# Basic teacher training
python main.py train-teacher

# Train teacher and evaluate
python main.py train-teacher --evaluate

# Full pipeline: train teacher, evaluate, and generate samples
python main.py train-teacher --evaluate --generate-samples --num-samples 5

# Train in background
nohup python train_teacher.py > teacher_train.log 2>&1 &
# Check logs
tail -f teacher_train.log
# Check process
ps -ef | grep train_teacher.py
```

The trained teacher model will be saved to `results_teacher/`.

### 1. Train the Student Model (Knowledge Distillation)

```bash
# Basic training
python main.py train

# Train and evaluate
python main.py train --evaluate

# Full pipeline: train, evaluate, and generate samples
python main.py train --evaluate --generate-samples --num-samples 5

# Train in background
nohup python distill.py > distill.log 2>&1 &
# Check logs
tail -f distill.log
# Check process
ps -ef | grep distill.py
```

### 2. Evaluate Models

Compute perplexity, BLEU, and F1 scores on test dataset:

```bash
python main.py evaluate
```

### 3. Test Model Outputs

Generate outputs and measure performance metrics:

```bash
# Test the distilled model
python main.py test

# Test with comparison to original model
python main.py test --compare-original

# Test with custom number of samples
python main.py test --num-samples 10 --compare-original

# Save results to file
python main.py test --compare-original --output-file test_results.json
```

Test metrics include:
- Output length (tokens)
- Generation time per sample
- Average generation time
- Speedup ratio (when comparing models)

### 4. Generate Samples

Generate text from the trained model:

```bash
# Generate with default prompts
python main.py generate

# Generate with custom prompts
python main.py generate --prompts "What is AI?" "Explain machine learning"

# Generate more samples
python main.py generate --num-samples 10
```

### 5. View Configuration

Display or save the current configuration:

```bash
# Display configuration
python main.py config

# Save configuration to file
python main.py config --save config_backup.json
```

### 6. Launch Web Interface

Launch the Gradio web interface for interactive model comparison:

```bash
python main.py gradio
```

### 7. Upload Models to HuggingFace Hub

If `push_to_hub` is enabled in config, models are automatically uploaded after training:

```bash
# Step 1: Authenticate with HuggingFace
huggingface-cli login

# Step 2: Create repositories on HuggingFace Hub
# - Create student model repo: https://huggingface.co/new
# - Create teacher model repo: https://huggingface.co/new

# Step 3: Update config.py with your repository names
# "hub": {
#     "push_to_hub": True,
#     "repo_name": "your-username/distil-logits-student",
#     "repo_name_teacher": "your-username/distil-logits-teacher"
# }

# Step 4: Run training - models will be pushed automatically after completion
python main.py train-teacher
python main.py train
```

**What gets uploaded:**
- Model weights and architecture
- Tokenizer configuration
- Training configuration
- README with model card information

**Troubleshooting:**
- If `push_to_hub=True` but `repo_name` is not specified, training will complete but upload will fail with an error message
- Make sure repositories exist on HuggingFace Hub before running training
- Ensure you have write access to the specified repositories

## Command Reference

### train-teacher
Train the teacher model using supervised fine-tuning (SFT).

**Options:**
- `--evaluate`: Run evaluation after training
- `--generate-samples`: Generate samples after training
- `--num-samples N`: Number of samples to generate (default: 5)

**Output:**
- Trained model saved to `results_teacher/`
- Periodic test evaluation during training
- Optional evaluation metrics and sample generation
- If `hub.push_to_hub=true`: Model automatically uploaded to HuggingFace Hub

**Example:**
```bash
python main.py train-teacher --evaluate --generate-samples --num-samples 10
```

### train
Train the distilled student model using knowledge distillation.

**Options:**
- `--evaluate`: Run evaluation after training
- `--generate-samples`: Generate samples after training
- `--num-samples N`: Number of samples to generate (default: 5)

**Output:**
- Trained model saved to `results/`
- Periodic test evaluation during training
- Optional evaluation metrics and sample generation
- If `hub.push_to_hub=true`: Model automatically uploaded to HuggingFace Hub

**Example:**
```bash
python main.py train --evaluate --generate-samples --num-samples 10
```

### evaluate
Evaluate original and distilled models on test dataset.

**Options:**
- `--max-samples N`: Maximum samples to evaluate (default: 500)

**Output:**
- Perplexity and loss for each model
- BLEU and F1 scores
- Comparison metrics
- Results saved to JSON file

**Example:**
```bash
python main.py evaluate --max-samples 1000
```

### test
Test model outputs with performance metrics.

**Options:**
- `--model-path PATH`: Path to model to test (default: ./results)
- `--compare-original`: Compare with original student model
- `--num-samples N`: Number of test samples (default: 10)
- `--output-file FILE`: Save results to JSON file

**Metrics:**
- Total outputs generated
- Average output length
- Average generation time
- Speedup ratio (when comparing)

**Example:**
```bash
python main.py test --compare-original --num-samples 20 --output-file test_results.json
```

### generate
Generate text samples from the trained model.

**Options:**
- `--model-path PATH`: Path to model (default: ./results)
- `--num-samples N`: Number of samples (default: 5)
- `--prompts TEXT...`: Custom prompts for generation

**Example:**
```bash
python main.py generate --num-samples 5 --prompts "What is AI?" "Explain ML"
```

### config
Display or save configuration.

**Options:**
- `--save FILE`: Save configuration to JSON file

**Example:**
```bash
python main.py config --save my_config.json
```

### gradio
Launch the Gradio web interface for interactive model comparison.

**Example:**
```bash
python main.py gradio
```

## Project Structure

```
distill_logits/
├── main.py                 # Main entry point with CLI
├── config.py               # Configuration settings
├── train_teacher.py        # Teacher model training script
├── distill.py        # Student model training with knowledge distillation
├── evaluate.py             # Evaluation functions (perplexity, BLEU, F1)
├── data_processing.py      # Dataset loading and preprocessing
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Key Components

### Teacher Model Training
The `train_teacher.py` script provides supervised fine-tuning (SFT) for the teacher model:
- Uses standard SFTTrainer from HuggingFace transformers
- Loads datasets with automatic caching
- Includes periodic test evaluation during training
- Saves trained model to `results_teacher/`
- Supports all standard training features (flash attention, mixed precision, etc.)

### LogitsTrainer
Custom trainer class that implements logits-based knowledge distillation:
- Computes KL-divergence loss between student and teacher logits
- Supports logit padding for vocabulary size mismatch
- Combines KD loss with original language modeling loss

### Evaluation Metrics
- **Perplexity**: Measures model's ability to predict test data
- **BLEU Score**: Evaluates n-gram overlap with reference text
- **F1 Score**: Token-level precision and recall

### Data Processing
- Supports FreedomIntelligence dataset format
- Automatic chat template formatting
- Tokenization with configurable max length
- Train/test split (90/10)

## Training Workflow

### Recommended Training Pipeline

1. **Train Teacher Model (Optional)**
   - Fine-tune the teacher model on your dataset using SFT
   - Command: `python main.py train-teacher`
   - Output: `results_teacher/`

2. **Train Student Model with Knowledge Distillation**
   - Train the student model using logits-based KD from the teacher
   - Command: `python main.py train`
   - Output: `results/`
   - The student model learns from both the teacher's logits and the original labels

3. **Evaluate and Test**
   - Evaluate metrics: `python main.py evaluate`
   - Test outputs: `python main.py test --compare-original`
   - Generate samples: `python main.py generate`

### Alternative: Direct Student Training
If you prefer to use a pre-trained teacher model without fine-tuning:
- Skip step 1 and directly run step 2
- The teacher model will be loaded as-is from HuggingFace

## Training Details

### Loss Function
```
Loss = α * KL_Divergence(student_logits, teacher_logits) + (1-α) * CrossEntropy(student_logits, labels)
```

Where:
- `α` (alpha): Balance between distillation and original loss
- `temperature`: Controls softness of probability distributions

### Optimization
- Optimizer: AdamW
- Learning rate scheduler: Cosine annealing
- Gradient accumulation: Supported
- Mixed precision: BF16 support
- Flash Attention 2: Optional for faster computation

## Output Files

### Teacher Model Training
- `results_teacher/checkpoint-*/`: Teacher model checkpoints
- `results_teacher/pytorch_model.bin`: Final trained teacher model
- `results_teacher/training_logs/`: Training logs and metrics

### Student Model Training
- `results/checkpoint-*/`: Student model checkpoints
- `results/pytorch_model.bin`: Final trained student model
- `results/training_logs/`: Training logs and metrics

### Evaluation
- `results/evaluation/evaluation_results_*.json`: Evaluation metrics and comparison

### Testing
- `test_results_*.json`: Test output metrics (auto-generated with timestamp if `--output-file` not specified)
