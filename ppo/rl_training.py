# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
import os
import dill
import sys

from tqdm.asyncio import tqdm_asyncio
from langchain.llms import OpenAIChat, OpenAI
from torch.nn.utils.rnn import pad_sequence
import asyncio
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline, BitsAndBytesConfig
import json
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import evaluate
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from trl import AutoModelForCausalLMWithValueHead
from datasets import load_dataset, concatenate_datasets
import openai
from evaluate import load
import subprocess
import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
tqdm.pandas()

TEST_CASE_PATH = "/convei_nas2/hjchae/code-editing/data/new_merge-100"


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    vllm_model_name: Optional[str] = field(default="/convei_nas2/hjchae/code-editing/checkpoints/CodeLlama-7b-Instruct-hf/code-50k-feedback-strict-filtered-feedback-only-checkpoint-5000-merged")



parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
reward_model_name = script_args.reward_model_name
#dataset_name = "lvwerra/stack-exchange-paired"
config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
    remove_unused_columns=False
)

# train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
# path = 'evaluation/Collie/sample_data/processed_task_specific_data'

# with open(os.path.join(path, script_args.constraint,"train.dill"), 'rb') as f:
#     train_dataset = dill.load(f)
# print(f"Train dataset size: {len(train_dataset)}")

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

class CollieDataset(Dataset):
    def __init__(self, data_list):
        self.data = [d for d in data_list if tokenizer(d['example'], return_tensors="pt")['input_ids'].squeeze(0).size()[-1] < script_args.output_max_length]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        #example = sample['example']
        #targets = sample['targets']
        # Convert the text data to tensors or any other preprocessing you'd like
        # For simplicity, let's just return the data as is.
        query = f"Question: {sample['prompt']}\n\nAnswer: "
        tokenized_query = tokenizer(query, truncation=True,return_tensors="pt")['input_ids'].squeeze(0)
        return {
            'idx': idx,
            'example': sample['example'],
            'targets': sample['targets'],
            'metadata': sample['metadata'],
            # 'constraint': sample['constraint'],
            # 'prompt': sample['prompt'],
            "input_ids":tokenized_query,
            "query": query
        }
    def get_constraint(self, idx):
        return self.data[idx]['constraint']
        
# train = CollieDataset(train_dataset)



# def collator(data):
#     return dict((key, [d[key] for d in data]) for key in data[0])

dataset = load_dataset("DLI-Lab/code-50k-feedback-strict-filtered")
raw_trainset = dataset['train'] 
raw_evalset = dataset['eval']

problem_ids_to_index = {pi:i for i, pi in enumerate(list(set(raw_trainset['problem_id']+raw_evalset['problem_id'])))}
print()
problem_index_to_id = {v:k for k,v in problem_ids_to_index.items()}
def format_dataset(dataset):
    instruction = "Provide feedback on the errors in the given code and suggest the correct code to address the described problem."
    original_columns = dataset.column_names
    def preprocess_function(examples):
        new_examples = {
            "input_ids": [],
            "problem_index": []
        }
        # print(examples)
        for i in range(len(examples['problem_id'])):
            input_string = f"{instruction}\nProblem Description:{examples['description'][i]}\nIncorrect Code:\n{examples['wrong_code'][i]}\n"
            input_ids = tokenizer(input_string, truncation=True,return_tensors="pt", max_length=2048)['input_ids'].squeeze(0)
            problem_id = examples['problem_id'][i]
            problem_index = problem_ids_to_index[problem_id]
            new_examples['input_ids'].append(input_ids)
            new_examples['problem_index'].append(torch.tensor([problem_index]))
        
        return new_examples

    # formatted_dataset = dataset.map(
    #     lambda x: {
    #         "input": f"{instruction}\nProblem Description:{x['description']}\nIncorrect Code:\n{x['wrong_code']}\n",
    #         "output": f"Feedback:\n{x['feedback']}\nCorrect code:\n{x['correct_code']}",
    #         "problem_id": x['problem_id'],
    #         "input_ids":
    #     }
    # )
    formatted_dataset = dataset.map(preprocess_function, batched=True, remove_columns=original_columns)
    formatted_dataset.set_format(type="torch")
    return formatted_dataset
def collator(data):
    # out_dict = {}
    # for key in data[0].keys():
    #     if key == "input_ids":
    #         out_dict[key] = pad_sequence([d[key] for d in data], batch_first=True, padding_value=tokenizer.pad_token_id)
    #     else:
    #         our_dict[key] = [d[key] for d in data]
    return dict((key, [d[key] for d in data]) for key in data[0])
    # return out_dict 

trainset = format_dataset(raw_trainset)

evalset = format_dataset(raw_evalset)
# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map={"": current_device},
    peft_config=lora_config,
)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=trainset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline using our reward model, passing the
# model name and the sentiment analysis pipeline arguments. Let's also make sure to
# set the device to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.95,
    "temperature": 0.4,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "max_new_tokens": script_args.output_max_length,
}
output_min_length = 1
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)
total_iterations = min(len(ppo_trainer.dataloader), config.total_ppo_epochs)

# unwrapped_model: "AutoModelForCausalLMWithValueHead" = ppo_trainer.accelerator.unwrap_model(model)

async def async_generate(llm, model_input):
    while True:
        try:
            response = await llm.agenerate(prompts=[model_input])  # if you need it
            # print("Completion result:", completion)
            break
        except Exception as e:
            print(f"Exception occurred: {e}")
            
            # response = None
            # return None

    result = response.generations[0][0].text
    return result
async def generate_concurrently(all_model_input, stop, port, args):
    # code generation model only
    llm = OpenAI(
        model_name=args.vllm_model_name,
        openai_api_base=f"http://localhost:{port}/v1",
        openai_api_key="EMPTY",
        max_tokens=1024,
        top_p=0.95,
        temperature=0.1,
    )
    tasks = [
        async_generate(llm, model_input) for i, model_input in enumerate(all_model_input)
    ]
    return await tqdm_asyncio.gather(*tasks)

def run_code(code, input_data):
    process = subprocess.Popen(
        ['python', '-c', code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Join inputs with newline and encode
    input_data = input_data.encode('utf-8')
    stdout_data, stderr_data = process.communicate(input=input_data)
    
    if process.returncode != 0:
        # There was an error
        print(code)
        print(Exception(f"Error executing code:\n{stderr_data.decode('utf-8')}"))
        return None
    
    return stdout_data.decode('utf-8')

def score_output(predictions, references):
    score = sum([p==r for p,r in zip(predictions, references)])/len(predictions)
    return torch.tensor([score])

def calculate_test_score(refined_codes, problem_ids):
    test_case_list = []
    for pi in problem_ids:
        with open(os.path.join(TEST_CASE_PATH, f"{pi}.json"),"r") as f:
            cur_test_case = json.load(f)['data'][0]['test_cases']
            test_case_list.append(cur_test_case)
    
    test_case_predictions = []
    scores = []
    for ci, c in enumerate(refined_codes): 
        cur_testcases = test_case_list[ci]
        cur_inputs = [t['input'] for t in cur_testcases]
        cur_outputs = [run_code(c, ti) for ti in cur_inputs]
        cur_references =[t['output'] for t in cur_testcases]
        cur_score = score_output(cur_outputs, cur_references)
        scores.append(cur_score)
    
    return scores 
    

async def generate_refinement(model_inputs, batch_feedback, args):
    # format model input for code refinement
    model_inputs_for_refinement = []
    for model_input, cur_feedback in zip(model_inputs,batch_feedback):
        new_model_input = f"{model_input}{cur_feedback}\nCorrect code:\n"
        model_inputs_for_refinement.append(new_model_input)
    refined_codes = await generate_concurrently(model_inputs_for_refinement, None, 8000,args)

    return refined_codes

def get_test_score_with_refine_generation(questions, batch_feedback, problem_ids, args):
    refined_codes = asyncio.run(generate_refinement(questions, batch_feedback, args))
    scores = calculate_test_score(refined_codes, problem_ids)
    scores = torch.Tensor(scores)
    return scores


with tqdm(enumerate(ppo_trainer.dataloader), total=total_iterations) as pbar:
    for epoch, batch in pbar:
        if epoch >= config.total_ppo_epochs:
            break
        
        # unwrapped_model.gradient_checkpointing_disable()
        # unwrapped_model.config.use_cache = True
        model.eval()
        # prompt = batch["prompt"]
        question_tensors = batch["input_ids"]
        problem_indices = batch['problem_index']
        problem_ids = [problem_index_to_id[int(pi[0])] for pi in problem_indices]

        # print(problem_ids)
        # import pdb
        # pdb.set_trace()
        # # print(len(question_tensors))
        # print(question_tensors.shape)
        print(question_tensors[0].shape)
        # print(question_tensors)
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        question_strings = tokenizer.batch_decode(question_tensors, skip_special_tokens=True)
        # Compute reward score (using the sentiment analysis pipeline)
        # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            
        # outputs = [train.get_constraint(i).check(r.split('Answer: ')[-1], t) for r, t, i in zip(batch['response'], batch['targets'], batch['idx'])]
        # rewards = [torch.tensor([float(o)]).to(question_tensors[0].device) for o in outputs]
        outputs = get_test_score_with_refine_generation(question_strings, batch['response'], problem_ids, script_args)
        rewards = [torch.tensor([float(o)]).to(question_tensors[0].device) for o in outputs]

        # unwrapped_model.gradient_checkpointing_enable()
        # unwrapped_model.config.use_cache = False
        model.train()
        for ri in range(len(response_tensors)):
            response_tensors[ri] = response_tensors[ri].to(question_tensors[0].device)
        # print("Questiuon Tensor")
        # print(question_tensors)
        # print("Response Tensor")
        # print(response_tensors)
        # print("Reward Tensor")
        # print(rewards)
        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
