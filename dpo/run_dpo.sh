#!/bin/bash
export WANDB_ENTITY='tutoring-convei'
export WANDB_PROJECT='code-dpo'

run_name="dpo-codellama2-only-feedback-6k-150_test"
sft_model_checkpoint="/convei_nas2/hjchae/dongjin/code-editing/qlora/output/codellama-2-7b-only-feedback-6k/checkpoint-150-merged"
config_file="dpo/config_dpo.yaml"
output_dir="dpo/checkpoints/${run_name}"
data_path="/convei_nas2/hjchae/code-editing/dpo/data"

accelerate launch \
    --config_file $config_file dpo/run_dpo.py \
    --model_name_or_path=${sft_model_checkpoint} \
    --data_dir=$data_path \
    --use_local 0 \
    --output_dir $output_dir \
    --logging_steps 5 \
    --max_steps 2000 \
    --save_steps 50 \
    --evaluation_strategy steps \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --max_length 2048 \
    --report_to wandb \
    --eval_steps 25 \
    --run_name $run_name 

