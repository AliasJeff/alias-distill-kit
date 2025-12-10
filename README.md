# alias-distill-kit

Comprehensive knowledge distillation framework for training efficient student models using combined logits and hidden state distillation from teacher models.

## Overview

alias-distill-kit is a comprehensive knowledge distillation toolkit that enables you to:
- Train efficient student models by distilling knowledge from larger teacher models
- Use combined logits and hidden state KL-divergence loss for effective knowledge transfer
- Evaluate model performance with multiple metrics (perplexity, BLEU, F1)
- Generate and test model outputs
- Compare original and distilled models

## Features

- **Dual-Level Knowledge Distillation**: Combines logits-based and hidden state-based KL-divergence loss with configurable weights
- **Multi-Layer Adaptation**: Proportional layer mapping between student and teacher models with learnable projections
- **Flexible Model Support**: Works with any HuggingFace transformer model with different architectures
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
python main.py test --num-samples 3 --compare-original

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
python main.py generate --num-samples 3
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

## Project Structure

```
alias-distill-kit/
├── src/
│   ├── README.md                          # Documentation
│   ├── config.py                          # Configuration settings
│   ├── data_processing.py                 # Dataset loading and preprocessing
│   ├── evaluate.py                        # Evaluation functions (perplexity, BLEU, F1)
│   ├── requirements.txt                   # Python dependencies
│   │
│   ├── distill_logits/
│   │   ├── main.py                        # Main entry point with CLI
│   │   ├── train_teacher.py               # Teacher model training script (SFT)
│   │   ├── distill.py                     # Combined logits + hidden state distillation trainer
│   │   └── __init__.py
│   │
│   └── distill_hidden/
│       ├── distil_hidden.py               # Hidden state distillation (reference implementation)
│       └── __init__.py
│
├── results/                               # Student model training outputs
│   ├── pytorch_model.bin                  # Trained student model
│   ├── adaptation_layer.pth               # Trained adaptation layer weights
│   └── training_logs/
│
├── results_teacher/                       # Teacher model training outputs
│   ├── pytorch_model.bin                  # Trained teacher model
│   └── training_logs/
│
└── .gitignore                             # Git ignore rules
```

### Key Directories

- **src/**: Main source code directory
  - **distill_logits/**: Combined logits and hidden state distillation implementation
  - **distill_hidden/**: Reference hidden state distillation implementation
- **results/**: Student model checkpoints and outputs
- **results_teacher/**: Teacher model checkpoints and outputs

## Training Details

### Loss Function
```
Loss = α * [w_logits * KL_logits + w_hidden * KL_hidden] + (1-α) * CrossEntropy(student_logits, labels)
```

Where:
- `α` (alpha): Balance between distillation and original loss (default: 0.5)
- `w_logits` (distillation_weight): Weight for logits-based KL divergence (default: 1.0)
- `w_hidden` (hidden_weight): Weight for hidden state-based KL divergence (default: 0.5)
- `temperature`: Controls softness of probability distributions (default: 3.0)

### Hidden State Distillation
- Student hidden states are projected to teacher dimensions using `MultiLayerAdaptationLayer`
- Each student layer is mapped to a proportional teacher layer based on relative position
- KL divergence is computed for each student-teacher layer pair
- Loss is averaged across all layer pairs and normalized by hidden dimension

### Logits Distillation
- KL divergence computed between temperature-scaled student and teacher logits
- Logits are padded to match vocabulary sizes if needed
- Loss is normalized by sequence length

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
- `results/adaptation_layer.pth`: Trained adaptation layer weights for hidden state distillation
- `results/training_logs/`: Training logs and metrics

### Evaluation
- `results/evaluation/evaluation_results_*.json`: Evaluation metrics and comparison

### Testing
- `test_results_*.json`: Test output metrics (auto-generated with timestamp if `--output-file` not specified)
