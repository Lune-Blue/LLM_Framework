
export OMP_NUM_THREADS=8
export WANDB_ENTITY='tutoring-convei'
export WANDB_PROJECT='coffee'

export PYTHONPATH="$PYTHONPATH:/home/taeyoon/nas2/newCodeEdit"

run_name="ours_and_oasst"
base_model_name="deepseek-ai/deepseek-coder-6.7b-instruct" #what model to use
dataset_name="ours_oasst" #what dataset to use
dataset_format="Chat" #what format to use --> wrirein qlora.py if-else statement
model_path_to_be_saved="checkpoints/${run_name}"


accelerate launch \
    --config_file scripts/accelerate_config.yaml train/main.py \
    --model_name_or_path $base_model_name \
    --chat_template_path chat_templates/ds.jinja \
    --use_auth \
    --output_dir ${model_path_to_be_saved} \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_steps 400 \
    --num_train_epochs 2 \
    --eval_dataset_size 512 \
    --max_eval_samples 1024 \
    --dataloader_num_workers 1 \
    --group_by_length True \
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
    --dataset $dataset_name \
    --target_max_len 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --adam_beta2 0.999 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 2024 \
    --dataset_format $dataset_format \
    --train_on_source False \
    --do_predict False \
    --report_to wandb \
    --run_name $run_name \
    --quantize 1 \
    # --full_finetune 1 
    
    
     
    # if you want qlora ->  erase --full_finetune 1
    # if you want quantize -> --quantize 1 --> bits and bytes
    #eval_dataset_size --> test size