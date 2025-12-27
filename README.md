- train-teacher qwen_afm_distill.yml -v
- distill qwen_afm_distill.yml -v
- infer qwen_afm_distill.yml --num-samples 3 --model-path outputs/models/qwen3-1.7b-teacher -v --max-new-tokens 512

bash run_all.sh
