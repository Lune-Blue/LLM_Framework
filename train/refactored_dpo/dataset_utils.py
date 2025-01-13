import os
import json
from typing import Dict, List
from datasets import Dataset, load_dataset

def get_stack_exchange_paired(dataset: str = "whatever", sanity_check: bool = False, cache_dir: str = None, num_proc=24, split: str = "train", tokenizer=None) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    if dataset.endswith(".json"):
        try:
            dataset = Dataset.from_json(dataset)
            print("Loaded local dataset")
        except:
            with open(dataset, "r") as f:
                data = json.load(f)
                if 'data' in data:
                    data = data['data']
            dataset = Dataset.from_dict(data)
            print("Loaded local dataset from dict")
    else:
        dataset = load_dataset(dataset, split=split, cache_dir=cache_dir)

    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    # Check if dataset matches the dpo_data format
    if not ("prompt" in dataset.column_names and "chosen" in dataset.column_names and "rejected" in dataset.column_names):
        def convert_to_dpo_format(samples) -> Dict[str, List[str]]:
            feedback_prompt_template = "Provide feedback on the errors in the given code.\nProblem Description: {description}\nIncorrect Code: {wrong_code}"
            feedback_answer_template = "Feedback: {feedback}"

            return {
                "prompt": [f"{tokenizer.bos_token}{feedback_prompt_template.format(description=desc, wrong_code=code)}" for desc, code in zip(samples["description"], samples["wrong_code"])],
                "chosen": [f"{feedback_answer_template.format(feedback=fb)}{tokenizer.eos_token}" for fb in samples["valuabe_feedback"]],
                "rejected": [f"{feedback_answer_template.format(feedback=fb)}{tokenizer.eos_token}" for fb in samples["invaluabe_feedback"]]
            }

        dataset = dataset.map(convert_to_dpo_format, batched=True, num_proc=num_proc, remove_columns=original_columns)
        print("Converted dataset to dpo_data format")

    return dataset

def count_tokens(tokenizer, text):
    return len(tokenizer(text)["input_ids"])
