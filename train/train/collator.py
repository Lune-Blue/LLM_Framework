import torch
from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import Dict, Sequence
from dataclasses import dataclass, field
from jinja2 import Template

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "</s>"

@dataclass
class DataCollatorBase:
    tokenizer: PreTrainedTokenizer
    target_max_len: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("This method needs to be implemented in subclasses.")

@dataclass
class DataCollatorForCausalLM(DataCollatorBase):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = []
        labels = []
        
        pad_token_id = self.tokenizer.eos_token_id
            
        for example in instances:
            input_ids_elem = [self.tokenizer.bos_token_id]
            label_mask = [IGNORE_INDEX]
            
            for msg in example["messages"]:
                role_tokens = self.tokenizer.encode(f"<|{msg['role']}|>\n", add_special_tokens=False)
                label_mask += [IGNORE_INDEX] * len(role_tokens)
                input_ids_elem += role_tokens

                if msg["role"] == "assistant":
                    content_tokens = self.tokenizer.encode(
                        msg["content"].strip() + self.tokenizer.eos_token + "\n", add_special_tokens=False
                    )
                    
                    
                    newline_len = len(self.tokenizer.encode("\n", add_special_tokens=False))
                    label_mask += content_tokens

                    assert content_tokens[-1-newline_len] == self.tokenizer.eos_token_id
                    for i in range(newline_len):
                        label_mask[-1-i] = IGNORE_INDEX
                    
                    label_mask += content_tokens
                    
                    try:
                        if content_tokens[-2] == self.tokenizer.eos_token_id:
                            for i in range(1):
                                label_mask[-1-i] = IGNORE_INDEX
                        elif content_tokens[-3] == self.tokenizer.eos_token_id:
                            for i in range(2):
                                label_mask[-1-i] = IGNORE_INDEX
                        else:
                            print('error on masking out the last "\n"')
                    except:
                        print('context length is smaller than 3??')
                    

                else:
                    content_tokens = self.tokenizer.encode(msg["content"].strip() + "\n", add_special_tokens=False)
                    label_mask += [IGNORE_INDEX] * len(content_tokens)
                    
                input_ids_elem += content_tokens
            
            if len(input_ids) <= self.target_max_len:
                pad_len = self.target_max_len - len(input_ids_elem)
                input_ids_elem += [pad_token_id] * pad_len
                label_mask += [IGNORE_INDEX] * pad_len
            
            else:
                input_ids_elem = input_ids_elem[:self.target_max_len]
                label_mask = label_mask[:self.target_max_len]
                input_ids_elem[-1] = self.tokenizer.eos_token_id
                label_mask[-1] = self.tokenizer.eos_token_id

            assert len(input_ids_elem) == len(label_mask)
            
            labels.append(label_mask)
            input_ids.append(input_ids_elem)

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        
        return {"input_ids": input_ids, "labels": labels}


@dataclass
class DataCollatorForDeepseekcoder(DataCollatorBase):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = []
        labels = []
        
        pad_token_id = self.tokenizer.eos_token_id

        for example in instances:
            tokenized_input, label_mask = self.format_input_and_labels(example["messages"])

            if len(tokenized_input) < self.target_max_len:
                pad_len = self.target_max_len - len(tokenized_input)
                tokenized_input += [pad_token_id] * pad_len
                label_mask += [IGNORE_INDEX] * pad_len
            else:
                tokenized_input = tokenized_input[:self.target_max_len]
                label_mask = label_mask[:self.target_max_len]
                tokenized_input[-1] = self.tokenizer.eos_token_id
                label_mask[-1] = IGNORE_INDEX

            # Ensure lengths are consistent
            assert len(tokenized_input) == self.target_max_len, f"Tokenized input length mismatch: {len(tokenized_input)} != {self.target_max_len}"
            assert len(label_mask) == self.target_max_len, f"Label mask length mismatch: {len(label_mask)} != {self.target_max_len}"

            input_ids.append(tokenized_input)
            labels.append(label_mask)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {"input_ids": input_ids, "labels": labels}

    def format_input_and_labels(self, messages):
        bos_token = self.tokenizer.bos_token if self.tokenizer.bos_token is not None else ""
        tokenized_input = []
        label_mask = []

        if bos_token:
            tokenized_bos = self.tokenizer.encode(bos_token, add_special_tokens=False)
            tokenized_input += tokenized_bos
            label_mask += [IGNORE_INDEX] * len(tokenized_bos)

        ns_found = False
        for message in messages:
            if message['role'] == 'system':
                ns_found = True

        if not ns_found:
            system_message = 'You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n'
            tokenized_system_message = self.tokenizer.encode(system_message, add_special_tokens=False)
            tokenized_input += tokenized_system_message
            label_mask += [IGNORE_INDEX] * len(tokenized_system_message)

        for message in messages:
            if message['role'] == 'system':
                message_content = message['content']
            elif message['role'] == 'user':
                message_content = f"### Instruction:\n{message['content']}\n"
            else:
                message_content = f"### Response:\n{message['content']}\n<|EOT|>\n"

            encoded_message = self.tokenizer.encode(message_content, add_special_tokens=False)
            message_length = len(encoded_message)
            
            tokenized_input += encoded_message
            if message['role'] == 'user' or message['role'] == 'system':
                label_mask += [IGNORE_INDEX] * message_length
            else:  # assistant role
                label_mask += encoded_message

        # Ensure lengths are consistent
        assert len(tokenized_input) == len(label_mask), f"Length mismatch: tokenized_input ({len(tokenized_input)}) vs label_mask ({len(label_mask)})"

        return tokenized_input, label_mask

if __name__ == "__main__":
    # Assuming we have a tokenizer object already created for Deepseekcoder
    tokenizer_deepseekcoder = tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")

    # Load the chat template from file
    with open('/home/taeyoon/nas2/newCodeEdit/chat_templates/ds.jinja', 'r') as file:
        chat_template = file.read()

    # Set the chat template
    tokenizer_deepseekcoder.chat_template = chat_template

    target_max_len = 128

    deepseekcoder_collator = DataCollatorForDeepseekcoder(tokenizer=tokenizer_deepseekcoder, target_max_len=target_max_len)

    # Instances would be a list of dictionaries with the appropriate data
    instances = [
        {
            "messages": [
                {"role": "system", "content": "System message."},
                {"role": "user", "content": "How do I sort a list in Python?"},
                {"role": "assistant", "content": "You can use the sorted() function or the sort() method."}
            ]
        }
        # Add more instances as needed
    ]

    # Collate data for Deepseekcoder model
    deepseekcoder_batch = deepseekcoder_collator(instances)

    # Print the collated batch to verify
    print(deepseekcoder_batch)