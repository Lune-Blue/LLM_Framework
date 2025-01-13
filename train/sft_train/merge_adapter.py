from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b","--base_model_name_or_path", type=str, required=True)
    parser.add_argument(
        "-p","--peft_model_path",
        type=str,
    )

    return parser.parse_args()


def main():
    args = get_args()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path, return_dict=True, torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    model = model.merge_and_unload()

    save_dir = args.peft_model_path + "-merged"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {args.peft_model_path}-merged")


if __name__ == "__main__":
    main()
