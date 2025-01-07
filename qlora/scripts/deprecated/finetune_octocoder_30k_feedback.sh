export WANDB_ENTITY='tutoring-convei'
export WANDB_PROJECT='code-edit'
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch \
    --config_file accelerate_config.yaml qlora.py \
    --model_name_or_path bigcode/octocoder \
    --use_auth \
    --output_dir ./new_output/octocoder-new-feedback-30k \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 50 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 1000 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset DLI-Lab/new-code-refined-annotated-30k \
    --source_max_len 1024 \
    --target_max_len 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --eval_steps 50 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --report_to wandb \
    --dataset_format octocoder_feedback \
    --train_on_source False \
    --do_predict True \
    # --predict_with_generate True \