#!/usr/bin/env bash
set -e

mkdir -p logs

echo "[1/4] train-teacher"
nohup train-teacher qwen_afm_distill.yml -v > logs/01_train_teacher.log 2>&1
wait

echo "[2/4] distill"
nohup distill qwen_afm_distill.yml -v > logs/02_distill.log 2>&1
wait

echo "[3/4] evaluate"
nohup evaluate qwen_afm_distill.yml -v > logs/03_evaluate.log 2>&1
wait

echo "[4/4] infer"
nohup infer qwen_afm_distill.yml \
  --num-samples 3 \
  --model-path outputs/models/qwen3-1.7b-teacher \
  -v --max-new-tokens 512 > logs/04_infer.log 2>&1 &
