# alias-distill-kit

A flexible and production-ready toolkit for knowledge distillation of large language models, supporting both online and offline distillation workflows with advanced logit compression.

## Features

- **Online Distillation**: Real-time teacher inference during student training
- **Offline Distillation**: Train from pre-captured teacher outputs with advanced compression
- **Advanced Logit Compression**: Novel polynomial approximation + quantization + bit-packing achieving vigorous compression ratios while preserving distillation quality
- **Flexible Loss Functions**: Composable losses including KL divergence, JSD, TVD, ranking losses, and hidden state alignment
- **Sparse & Dense Support**: Efficient sparse distributions (top-k) or exact dense distributions
- **Battle-tested**: The infrastructure powering Arcee's distilled model releases
- **HuggingFace Integration**: Built on Transformers, TRL, and Accelerate

## Installation

```bash
pip install -e .
```

### Optional: Logit Capture

To capture your own teacher outputs, install the capture dependencies:

```bash
pip install -e ".[capture]"
```

For most users, we recommend starting with the pre-captured teacher datasets we provide (see [Datasets](#datasets) below).

## Quick Start

The entire workflow is controlled via a configuration file (e.g., `qwen_afm_distill.yml`). You can run steps individually or as a complete pipeline.

### Option 1: Step-by-Step Execution (Interactive)

Run these commands to see logs in your terminal immediately.

**1. Train Teacher**

Bash

```
train-teacher qwen_afm_distill.yml -v
```

**2. Distill (Train Student)** Performs the knowledge distillation process.

Bash

```
distill qwen_afm_distill.yml -v
```

**3. Evaluate** Assess the performance of the distilled model.

Bash

```
evaluate qwen_afm_distill.yml -v
```

**4. Inference** Generate samples from the trained student model.

Bash

```
infer qwen_afm_distill.yml \
    --num-samples 3 \
    --model-path outputs/models/qwen3-1.7b-student \
    --max-new-tokens 2048 \
    -v
```

------

### Option 2: Background Execution (Nohup)

Use these commands to run tasks in the background and save logs to the `logs/` directory.

**Individual Steps:**

Bash

```
mkdir -p logs

# 1. Train Teacher
nohup train-teacher qwen_afm_distill.yml -v > logs/01_train_teacher.log 2>&1 &

# 2. Distill
nohup distill qwen_afm_distill.yml -v > logs/02_distill.log 2>&1 &

# 3. Evaluate
nohup evaluate qwen_afm_distill.yml -v > logs/03_evaluate.log 2>&1 &

# 4. Inference
nohup infer qwen_afm_distill.yml \
    --num-samples 3 \
    --model-path outputs/models/qwen3-1.7b-student \
    --max-new-tokens 2048 \
    -v > logs/04_infer.log 2>&1 &
```

**Run All (Automated Script):** Execute the entire pipeline sequentially.

Bash

```
nohup ./run_all.sh > logs/run_all.log 2>&1 &
```

For online distillation where the teacher runs alongside student training, see [`examples/afm_test.yml`](examples/afm_test.yml) for a complete configuration example.

## Core Concepts

### Knowledge Distillation for LLMs

Knowledge distillation transfers knowledge from a (potentially larger) "teacher" model to a "student" model. Instead of training only on hard labels (the correct token), the student learns from the teacher's probability distribution over tokens, which is a much richer learning signal.

**Key benefits:**
- Smaller, faster models with competitive performance
- Lower inference costs
- Easier deployment in resource-constrained environments

### Online vs Offline Distillation

**Online Distillation:**
- Teacher runs in real-time during student training
- No storage overhead
- Best when: You have sufficient VRAM for both models and dense distributions

**Offline Distillation:**
- Teacher outputs pre-captured and compressed
- Enables training multiple students from the same teacher
- Best when: VRAM-limited, reusing teacher signals, or training at large scale

### Sparse vs Dense Distributions

**Dense distributions** include probabilities for the full vocabulary. This is more accurate but memory-intensive.

**Sparse distributions** store only the top-k tokens and serve as a lossy, but useful and efficient, approximation of the full dense distribution. With sufficient training data, sparse distillation can achieve equivalent performance to dense.

Our kit supports both, with automatic chunking for memory-efficient processing of long sequences.

### Logit Compression

Our compression system balances storage efficiency with distillation quality:

1. Select top-k logits from teacher output
2. Sort by log-probability, optionally apply delta encoding
3. Fit polynomial to the distribution curve
4. Quantize residuals, with optional error diffusion
5. Bitpack everything into byte vectors
