import os
import argparse
from transformers import HfArgumentParser
from arguments import ScriptArguments, TrainingArgumentsClass
from dataset_utils import get_stack_exchange_paired
from model_utils import load_model
from training import train_model

def main(args, extra_args):
    model, model_ref, tokenizer = load_model(args)
    
    train_dataset = get_stack_exchange_paired(dataset=args.dataset, sanity_check=args.sanity_check, split="train", tokenizer=tokenizer)
    train_dataset = train_dataset.filter(lambda x: count_tokens(tokenizer, x["prompt"] + x["chosen"]) <= args.max_length and count_tokens(tokenizer, x["prompt"] + x["rejected"]) <= args.max_length)
    
    print("Train dataset length: ", len(train_dataset))
    
    eval_dataset = get_stack_exchange_paired(dataset=args.dataset, sanity_check=True, split="eval", tokenizer=tokenizer)
    eval_dataset = eval_dataset.filter(lambda x: count_tokens(tokenizer, x["prompt"] + x["chosen"]) <= args.max_length and count_tokens(tokenizer, x["prompt"] + x["rejected"]) <= args.max_length)
    
    print("Eval dataset length: ", len(eval_dataset))
    print(f"Train dataset example: {train_dataset[0]}")
    
    train_model(args, train_dataset, eval_dataset, model, model_ref, tokenizer)

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArgumentsClass))
    script_args, training_args, extra_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    script_args = argparse.Namespace(**vars(script_args), **vars(training_args))
    print(script_args)
    main(script_args, extra_args)
