export OMP_NUM_THREADS=8
export WANDB_ENTITY='tutoring-convei'
export WANDB_PROJECT='code-edit'
dataset_name=$1
dataset_format=$2
base_model_name="/convei_nas2/taeyoon/newCodeEdit/checkpoints/only_ours/checkpoint-9068/adapter_model-merged"
model_last_name=$(basename $base_model_name)
model_path_to_be_saved="checkpoints/rejection_sampling_2/${model_last_name}/${dataset_name}-${dataset_format}-scott-level-2"
export WANDB_NAME=$(basename $model_path_to_be_saved)
accelerate launch \
    --config_file qlora/accelerate_config.yaml qlora/qlora.py \
    --model_name_or_path $base_model_name \
    --use_auth \
    --output_dir ${model_path_to_be_saved} \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 100 \
    --save_total_limit 100 \
    --evaluation_strategy steps \
    --eval_dataset_size 2048 \
    --max_eval_samples 2000 \
    --per_device_eval_batch_size 4 \
    --max_new_tokens 2048 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --max_steps 50000 \
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
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --dataset DLI-Lab/$dataset_name \
    --source_max_len 2048 \
    --target_max_len 2048 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --eval_steps 100 \
    --learning_rate 2e-4 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --report_to wandb \
    --dataset_format $dataset_format \
    --train_on_source False \
    --do_predict False \
    # --predict_with_generate True \