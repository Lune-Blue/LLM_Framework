from transformers import TrainingArguments
from trl import DPOTrainer
from dataset_utils import count_tokens

def train_model(args, train_dataset, eval_dataset, model, model_ref, tokenizer):
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        remove_unused_columns=False,
        run_name=args.run_name,
    )

    if args.qlora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out", "wte"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    dpo_trainer = DPOTrainer(
        model=model,
        model_ref=model_ref,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
    )

    dpo_trainer.train()
    dpo_trainer.save_model(args.output_dir)

    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
