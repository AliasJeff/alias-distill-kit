- train-teacher qwen_afm_distill.yml -v
- distill qwen_afm_distill.yml -v
- infer qwen_afm_distill.yml --num-samples 3 --model-path outputs/models/qwen3-0.6b-student -v --max-new-tokens 512

nohup ./run_all.sh > logs/run_all.log 2>&1 &
