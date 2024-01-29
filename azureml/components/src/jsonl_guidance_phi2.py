import argparse
import importlib.util
import json
import pathlib

from typing import Any, Callable, Dict

import guidance

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aether_utils.jsonl_utils import line_map
from aether_utils.logging_utils import get_standard_logger_for_file


_logger = get_standard_logger_for_file(__file__)

USER_MODULE = "user_module"
GUIDANCE_FUNCTION = "guidance_generation"


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)

    # Information about the datasets
    datasets_group = parser.add_argument_group("Datasets")
    datasets_group.add_argument("--input_dataset", type=pathlib.Path, required=True)
    datasets_group.add_argument("--input_encoding", type=str, required=True)
    datasets_group.add_argument("--output_dataset", type=pathlib.Path, required=True)
    datasets_group.add_argument("--output_encoding", type=str, required=True)
    datasets_group.add_argument("--error_dataset", type=pathlib.Path, required=True)
    datasets_group.add_argument("--error_encoding", type=str, required=True)
    datasets_group.add_argument(
        "--common_dataset", type=pathlib.Path, required=False, default=None
    )
    datasets_group.add_argument("--common_encoding", type=str, required=False)

    # Information about the guidance program
    parser.add_argument("--guidance_program", type=pathlib.Path, required=True)
    parser.add_argument("--max_errors", type=int, required=True)

    args = parser.parse_args()
    return args

    
def get_guidance_function(
    program_path: pathlib.Path,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    _logger.debug("Importing guidance file")
    spec = importlib.util.spec_from_file_location(USER_MODULE, program_path)
    module_definition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_definition)

    guidance_func = getattr(module_definition, GUIDANCE_FUNCTION)

    return guidance_func

def main():
    args = parse_args()

    # Load the common data (if required)
    common_data = None
    if args.common_dataset is not None:
        _logger.info("Loading common dataset")
        with open(args.common_dataset, "r", encoding=args.common_encoding) as jf:
            common_data = json.load(jf)
    else:
        _logger.info("No common dataset present")


    torch.set_default_device("cuda")

    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

    inputs = tokenizer('''def print_prime(n):
    """
    Print all primes between 1 and n
    """''', return_tensors="pt", return_attention_mask=False)

    outputs = model.generate(**inputs, max_length=200)
    text = tokenizer.batch_decode(outputs)[0]
    print(text)

    _logger.info("Complete")


if __name__ == "__main__":
    main()
