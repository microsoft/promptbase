import argparse
import functools
import importlib.util
import pathlib
import sys

from typing import Any, Callable, Dict

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

import guidance

from shared.jsonl_utils import line_map
from shared.logging_utils import get_standard_logger_for_file

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

    # Information about the guidance program
    parser.add_argument("--guidance_program", type=pathlib.Path, required=True)

    # Information about the model
    model_group = parser.add_argument_group("Model Endpoint")
    model_group.add_argument("--azure_openai_endpoint", type=str, required=True)
    model_group.add_argument("--azure_openai_deployed_model", type=str, required=True)

    args = parser.parse_args()
    return args


def get_guidance_function(
    program_path: pathlib.Path,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    _logger.info("Importing guidance file")
    spec = importlib.util.spec_from_file_location(USER_MODULE, program_path)
    module_definition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_definition)

    guidance_func = getattr(module_definition, GUIDANCE_FUNCTION)

    return guidance_func


def get_model(
    endpoint: str,
    model: str,
) -> guidance.models.Model:
    _logger.info("Attempting to create model object")
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    assert token_provider is not None

    # Pending a fix going into the released version of guidance,
    # we can only work with chat models
    azureai_model = guidance.models.AzureOpenAIChat(
        model=model,
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
    )

    return azureai_model


def process_item(
    item: Dict[str, Any],
    guidance_function: Callable[[Any, Dict[str, Any]], Dict[str, Any]],
    language_model: guidance.models.Model,
) -> Dict[str, Any]:
    _logger.info(f"process_item: {item}")

    result = guidance_function(language_model, item)
    _logger.info(f"Checking keys")
    for k in result.keys():
        assert k not in item, f"Duplicate key: {k}"

    _logger.info(f"Updating item")
    item.update(**result)

    return item


def main():
    args = parse_args()

    # Get the function
    guidance_func = get_guidance_function(args.guidance_program)

    # Get the language model
    llm = get_model(
        endpoint=args.azure_openai_endpoint, model=args.azure_openai_deployed_model
    )

    # Bind them together
    processor = functools.partial(
        process_item, guidance_function=guidance_func, language_model=llm
    )

    # Run the processing
    line_map(
        map_func=processor,
        source_file=args.input_dataset,
        dest_file=args.output_dataset,
        source_encoding=args.input_encoding,
        dest_encoding=args.output_encoding,
    )

    _logger.info("Complete")


if __name__ == "__main__":
    main()
