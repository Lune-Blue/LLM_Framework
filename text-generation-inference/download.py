from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--local_dir", type=str, default=".")
    args = parser.parse_args()
    return args


def download(model_id: str, local_dir: str):
    snapshot_download(repo_id=model_id, local_dir=local_dir)


if __name__ == "__main__":
    args = parse_args()
    download(args.model_name, args.local_dir)
    print("Done")
