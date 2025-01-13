import os
import torch
import gc
import argparse
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Trainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
)
from config import ModelArguments, DataArguments, TrainingArguments, GenerationArguments
from model_utils import get_accelerate_model, print_trainable_parameters
from data_utils import make_data_module, get_last_checkpoint
from callbacks import SavePeftModelCallback
import logging
from jinja2 import Template

logger = logging.getLogger(__name__)

def train():
    torch.cuda.empty_cache()
    gc.collect()
    
    hfparser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GenerationArguments))
    model_args, data_args, training_args, generation_args, extra_args = hfparser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))
    print(args)

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print("Detected that training was already completed!")

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    
    ## set chat_template
    if model_args.chat_template_path:
        with open(model_args.chat_template_path, 'r') as f:
            chat_template = f.read()
        tokenizer.chat_template = chat_template
    
    
    model.config.use_cache = False
    print("loaded model")
    
    
    set_seed(args.seed)
    data_module = make_data_module(tokenizer=tokenizer, args=args)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    all_metrics = {"run_name": args.run_name}
    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module["predict_dataset"], metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        with open(os.path.join(args.output_dir, "predictions.jsonl"), "w") as fout:
            for i, example in enumerate(data_module["predict_dataset"]):
                example["prediction_with_input"] = predictions[i].strip()
                example["prediction"] = predictions[i].replace(example["input"], "").strip()
                fout.write(json.dumps(example) + "\n")
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if args.do_train or args.do_eval or args.do_predict:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    train()
