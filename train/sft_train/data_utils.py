import os
import pandas as pd
import torch
from datasets import load_dataset, Dataset, DatasetDict
from collator import DataCollatorForCausalLM, DataCollatorForDeepseekcoder

def local_dataset(dataset_name):
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith(".csv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith(".tsv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter="\t"))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset

def make_data_module(tokenizer, args):
    def load_data(dataset_name):
        if dataset_name == 'CHAT_EXAMPLE':
            full_dataset = local_dataset('CHAT_EXAMPLE.json')
            return full_dataset
        elif dataset_name == 'allenai/tulu-v2-sft-mixture':
            full_dataset = load_dataset('allenai/tulu-v2-sft-mixture')
            selected_dataset = full_dataset["train"].shuffle(seed=42)
            selected_dict = {"train": selected_dataset}
            return DatasetDict(selected_dict)
        elif dataset_name == 'tulu-v2-sft-mixture_gsm8k_mixed':
            full_dataset = local_dataset('dataset/gsm_tulu2_mix.json').shuffle(seed=42)
        elif dataset_name == 'MathInstruct_Cot':
            full_dataset = local_dataset('data/MathInstruct_COT_67804.json').shuffle(seed=42)
        elif dataset_name == 'ours':
            full_dataset = local_dataset('data/ours_train.json').shuffle(seed=42)
        elif dataset_name == "ours_oasst":
            full_dataset = local_dataset('data/ours_oasst_train.json').shuffle(seed=42)
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")
        return full_dataset

    def format_dataset(dataset):
        dataset = dataset.remove_columns([col for col in dataset.column_names["train"] if col not in ["messages"]])
        return dataset

    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset)
    print(dataset)
    
    if args.do_eval or args.do_predict:
        if "eval" in dataset:
            eval_dataset = dataset["eval"]
        else:
            print("Splitting train dataset in train and validation according to `eval_dataset_size`")
            dataset = dataset["train"].train_test_split(test_size=args.eval_dataset_size, shuffle=True, seed=42)
            eval_dataset = dataset["test"]
            
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
            
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {"length": sum(len(i['content']) for i in x['messages'])})
            
    if args.do_train:
        train_dataset = dataset["train"]
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {"length": sum(len(i['content']) for i in x['messages'])})
            
    data_collator = DataCollatorForDeepseekcoder(
        tokenizer=tokenizer,
        target_max_len=args.target_max_len,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator,
    )

def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, "completed"))
        if is_completed:
            return None, True
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith("checkpoint"):
                max_step = max(max_step, int(filename.replace("checkpoint-", "")))
        if max_step == 0:
            return None, is_completed
        checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint-{max_step}")
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed
    return None, False
