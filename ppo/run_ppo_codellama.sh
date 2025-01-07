#!/bin/bash
sft_model_checkpoint="/convei_nas2/hjchae/code-editing/checkpoints/CodeLlama-7b-Instruct-hf/code-50k-feedback-strict-filtered-feedback-only-checkpoint-5000-merged"
output_path="checkpoints/ppo/codellama-7b-50k-strict-filtered"
main_process_port=29501
refine_model="/convei_nas2/hjchae/code-editing/checkpoints/CodeLlama-7b-Instruct-hf/code-50k-feedback-strict-filtered-codegen-only-checkpoint-2300-merged"

# vLLM server
vllm_port=8000
SERVER_LOG_FILE="/convei_nas2/hjchae/code-editing/ppo/server.log"
# echo "Starting server..."
# CUDA_VISIBLE_DEVICES="6,7" python -m vllm.entrypoints.openai.api_server \
# --model $refine_model \
# --tensor-parallel-size 2 \
# --seed 42 \
# --port $vllm_port > ${SERVER_LOG_FILE} 2>&1 &

export WANDB_NAME="ppo-codellama"
export WANDB_ENTITY='tutoring-convei'
export WANDB_PROJECT="code-ppo"
accelerate launch --multi_gpu --num_machines 1  --num_processes 6 --main_process_port $main_process_port --config_file ppo/defualt_config.yaml \
ppo/rl_training.py \
    --log_with=wandb \
    --tokenizer_name=$sft_model_checkpoint \
    --model_name=$sft_model_checkpoint \
    --adafactor=False \
    --save_freq=50 \
    --output_max_length=256 \
    --batch_size=1 \
    --gradient_accumulation_steps=2 \
    --batched_gen=True \
    --ppo_epochs=4 \
    --seed=0 \
    --learning_rate=1.4e-5 \
    --early_stopping=True \
    --output_dir=$output_path \
    --vllm_model_name $refine_model
# python ppo/rl_training.py \