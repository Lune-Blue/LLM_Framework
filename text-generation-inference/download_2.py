from transformers import AutoModel, AutoTokenizer
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--save_dir", type=str, default=".")
    args = parser.parse_args()
    return args


def download_model(model_name):
    # Load pre-trained model and tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    args = parse_args()
    download_model(args.model_name)
    print("Done")
