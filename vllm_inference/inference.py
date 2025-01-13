import argparse
import asyncio
import json
import os
import random
import glob
import yaml
import copy
import re
from transformers.trainer_utils import set_seed
from prompters import AlpacaPrompter, PromptStyle
import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAIChat, OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from datasets import load_dataset, concatenate_datasets
import openai
from evaluate import load
import copy

import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
# openai.api_key = "EMPTY"
# openai.api_base = "http://localhost:8000/v1"
set_seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback_model_name", type=str, default=None)
    parser.add_argument("--generation_model_name", type=str, required=True)
    parser.add_argument("--feedback_model_port", type=int, default=None)
    parser.add_argument("--generation_model_port", type=int, required=True)
    parser.add_argument("--data_name", type=str, default="DLI-Lab/med_critic1_train_1000")
    parser.add_argument("--prompt", type=str, default="prompt/feedback_inference_test.yaml")
    parser.add_argument("--prompt_key", type=str, default="base")
    parser.add_argument("--use_feedback", type=str, default="Yes", choices=["Yes", "No"])
    # parser.add_argument("--do_iterate", default="Yes", choices=["Yes", "No"])
    # parser.add_argument("--iterate_num", type=int, default=5, help="number of iteration")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--num_try", type=int, default=3, help="number of samples to generate")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--limit", type=int, default=100)
    # parser.add_argument("--min_context_len", type=int, default=0)
    return parser.parse_args()


def load_data(data_name, split=None):
    data = load_dataset(data_name)
    print("=========== dataset statistics ===========")
    print(len(data[split]))
    print("==========================================")
    if args.limit:
        return [data[split][i] for i in range(args.limit)]
    return data[split]


def load_json(input_file):
    with open(input_file, "r") as f:
        data = json.load(f)
    return data


def load_prompt(prompt_path):
    with open(args.prompt, "r", encoding="UTF-8") as f:
        prompt = yaml.load(f, Loader=yaml.FullLoader)[args.prompt_key]
    return prompt

def prepare_model_input(prompter, input_data):
    get_input = input_data["input"]
    get_instruction = input_data["instruction"]
    get_output = ""
    model_input = prompter.build_prompt(instruction=get_instruction, input=get_input, output=get_output)
    return model_input


def load_and_prepare_data(args):
    dataset = load_data(args.data_name, args.split)
    prompter = AlpacaPrompter()
    all_model_inputs = []
    print("### load and prepare data")
    for data in tqdm(dataset):
        model_input = prepare_model_input(prompter, data)
        all_model_inputs.append([model_input, data])
    return all_model_inputs


async def async_generate(llm, model_input, idx, save_dir):
    while True:
        try:
            response = await llm.agenerate(prompts=[model_input[0]])  # if you need it
            # print("Completion result:", completion)
            break
        except Exception as e:
            print(f"Exception occurred: {e}")
            # response = None
            # return None

    result = {
        "prediction": response.generations[0][0].text,
        "model_input": model_input[0],
        **model_input[1],
    }

    return result


async def generate_concurrently(all_model_input, start_idx, stop, args, feedback_flag):
    if feedback_flag:  ## if feedback model
        llm = OpenAI(
            model_name=args.feedback_model_name,
            openai_api_base=f"http://localhost:{args.feedback_model_port}/v1",
            openai_api_key="EMPTY",
            max_tokens=128,
            top_p=0.95,
            temperature=0.5,
            frequency_penalty=0.4,
            stop=stop,
        )
    else:
        llm = OpenAI(
            model_name=args.generation_model_name,
            openai_api_base=f"http://localhost:{args.generation_model_port}/v1",
            openai_api_key="EMPTY",
            max_tokens=1024,
            top_p=0.95,
            temperature=0.1,
            frequency_penalty=0.0,
            stop=stop,
        )
    tasks = [
        async_generate(llm, model_input, i + start_idx, args.save_dir) for i, model_input in enumerate(all_model_input)
    ]
    return await tqdm_asyncio.gather(*tasks)



def prepare_correction_input(all_model_inputs, all_feedback_results):
    all_correction_model_inputs = []
    for i in range(len(all_model_inputs)):
        model_input_dict = copy.deepcopy(all_model_inputs[i])
        model_input = model_input_dict[0]
        cur_feedback = all_feedback_results[i]["prediction"]
        model_input_dict[1]["feedback"] = cur_feedback.split("\n")
        new_model_input = f"{model_input}{cur_feedback}\n"
        model_input_dict[0] = new_model_input
        all_correction_model_inputs.append(model_input_dict)

    return all_correction_model_inputs



async def main(args):
    all_model_inputs = load_and_prepare_data(args)
    for i in range(args.num_try):
        save_path = os.path.join(args.save_dir, f"{i+1}_sample")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if args.use_feedback == "Yes":
            if args.feedback_model_name is None:  ## if use feedback but feedback model is not defined
                args.feedback_model_name = args.generation_model_name
            if args.feedback_model_port is None:
                args.feedback_model_port = args.generation_model_port
            all_feedback_results = await generate_concurrently(
                all_model_inputs, 0, None, args, True
            )  ## generate feedback
            all_correction_input = prepare_correction_input(all_model_inputs, all_feedback_results)  # Correction
            all_results = await generate_concurrently(all_correction_input, 0, None, args, False)  # generate code
        else:
            all_results = await generate_concurrently(all_model_inputs, 0, None, args, False)  # generate code

        # reformated_eval_results = reformat_data(eval_results)
        with open(os.path.join(save_path, "seed_try.json"), "w", encoding="UTF-8") as f:
            json.dump(all_results, f, indent=4)



if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
    print("Done!")
