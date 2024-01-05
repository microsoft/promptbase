import argparse
import functools
import importlib.util
import json
import pathlib

from typing import Any, Callable, Dict

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

import guidance

from shared.jsonl_utils_multiprocessing import line_map_mp
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
    datasets_group.add_argument("--error_dataset", type=pathlib.Path, required=True)
    datasets_group.add_argument("--error_encoding", type=str, required=True)
    datasets_group.add_argument(
        "--common_dataset", type=pathlib.Path, required=False, default=None
    )
    datasets_group.add_argument("--common_encoding", type=str, required=False)

    # Information about the guidance program
    parser.add_argument("--guidance_program", type=pathlib.Path, required=True)
    parser.add_argument("--guidance_workers", type=int, required=True)
    parser.add_argument("--max_errors", type=int, required=True)

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
    program_path: pathlib.Path,
    endpoint: str,
    model: str,
    common_data: Any | None,
) -> Dict[str, Any]:
    _logger.info(f"process_item: {item}")

    guidance_function = get_guidance_function(program_path)
    language_model = get_model(endpoint, model)
    result = guidance_function(language_model, item, common=common_data)
    _logger.info(f"Checking keys")
    for k in result.keys():
        assert k not in item, f"Duplicate key: {k}"

    _logger.info(f"Updating item")
    item.update(**result)

    return item


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

    # Bind arguments to the processor function
    processor = functools.partial(
        process_item,
        program_path=args.guidance_program,
        endpoint=args.azure_openai_endpoint,
        model=args.azure_openai_deployed_model,
        common_data=common_data,
    )

    # Run the processing
    line_map_mp(
        map_func=processor,
        source_file=args.input_dataset,
        dest_file=args.output_dataset,
        source_encoding=args.input_encoding,
        dest_encoding=args.output_encoding,
        error_file=args.error_dataset,
        error_encoding=args.error_encoding,
        n_worker_tasks=args.guidance_workers,
        n_errors_max=args.max_errors,
    )

    _logger.info("Complete")


if __name__ == "__main__":
    main()
