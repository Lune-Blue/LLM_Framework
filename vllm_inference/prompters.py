import logging
from enum import Enum
from typing import Generator, Optional, Union

from colorama import Fore

IGNORE_TOKEN_ID = -100
REPR_TEMPLATE = "\n<start>\n" + Fore.CYAN + "{full_prompt}" + Fore.RESET + "\n<end>\n"


class PromptStyle(Enum):
    """
    Enum for prompt styles
    """

    INSTRUCT = "instruct"
    CHAT = "chat"
    CHATML = "chatml"
    PHI = "phi"


class Prompter:
    """
    Base prompter class for all prompters
    """


class AlpacaPrompter(Prompter):
    """
    Base class for alpaca prompters
    """

    system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    system_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    system_format: str = "{system}"
    turn_format: str
    turn_no_input_format: str
    prompt_style: Optional[str] = None

    def __init__(self, prompt_style: Optional[str] = PromptStyle.INSTRUCT.value):
        self.prompt_style = prompt_style if prompt_style else PromptStyle.INSTRUCT.value
        self.match_prompt_style()

    def match_prompt_style(self):
        # pylint: disable=duplicate-code
        if self.prompt_style == PromptStyle.INSTRUCT.value:
            self.turn_format = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            self.turn_no_input_format = (
                "### Instruction:\n{instruction}\n\n### Response:\n"
            )
            self.system_format = "{system}\n\n"
        elif self.prompt_style == PromptStyle.CHAT.value:
            self.turn_format = "USER: {instruction}\n{input}\nASSISTANT:"
            self.turn_no_input_format = "USER: {instruction}\nASSISTANT:"
            self.system_format = "SYSTEM: {system}\n"
        elif self.prompt_style == PromptStyle.CHATML.value:
            self.turn_format = "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n<|im_start|>assistant\n"
            self.turn_no_input_format = (
                "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            )
            self.system_format = "<|im_start|>system\n{system}<|im_end|>\n"
        elif self.prompt_style == PromptStyle.PHI.value:
            self.turn_format = "<|user|>\n{instruction}<|end|>{input}<|assistant|>"
            self.turn_no_input_format = (
                "<|user|>\n{instruction}<|end|>\n<|assistant|>\n"
            )
            self.system_format = "<|system|>\n{system}<|end|>\n"

    def _build_result(self, instruction, input_text, output):
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input_text:
            res = (
                self.system_format.format(system=self.system_prompt)
                if self.system_prompt
                else ""
            ) + self.turn_format.format(instruction=instruction, input=input_text)
        else:
            res = (
                self.system_format.format(system=self.system_no_input_prompt)
                if self.system_no_input_prompt
                else ""
            ) + self.turn_no_input_format.format(instruction=instruction)
        if output:
            res = f"{res}{output}"

        return res

    def build_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,  # pylint: disable=redefined-builtin
        output: Union[None, str] = None,
    ) -> str:
        return self._build_result(instruction, input, output)

    def __repr__(self) -> str:
        return REPR_TEMPLATE.format(
            full_prompt=self._build_result("{instruction}", "{input}", "{output}")
        )