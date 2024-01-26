import argparse
import importlib.util
import json
import pathlib

from typing import Any, Callable, Dict

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

import guidance

from aether_utils.jsonl_utils_multiprocessing import line_map_mp, ItemMapper
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
    _logger.debug("Importing guidance file")
    spec = importlib.util.spec_from_file_location(USER_MODULE, program_path)
    module_definition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_definition)

    guidance_func = getattr(module_definition, GUIDANCE_FUNCTION)

    return guidance_func


def get_model(
    endpoint: str,
    model: str,
) -> guidance.models.Model:
    _logger.debug("Attempting to create model object")
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


class GuidanceAzureML(ItemMapper):
    def __init__(
        self, program_path: pathlib.Path, endpoint: str, model: str, common_data: any
    ):
        super().__init__()
        self._program_path = program_path
        self._endpoint = endpoint
        self._model = model
        self._common_data = common_data

    def start_up(self, worker_id: int) -> None:
        _logger.info(f"Starting up {worker_id}")
        self._guidance_function = get_guidance_function(self._program_path)
        self._azure_credential = DefaultAzureCredential()

    def _get_model(self) -> guidance.models.Model:
        token_provider = get_bearer_token_provider(
            self._azure_credential, "https://cognitiveservices.azure.com/.default"
        )
        assert token_provider is not None

        # Pending a fix going into the released version of guidance,
        # we can only work with chat models
        azureai_model = guidance.models.AzureOpenAIChat(
            model=self._model,
            azure_endpoint=self._endpoint,
            azure_ad_token_provider=token_provider,
        )

        return azureai_model

    def map(self, item: dict[str, any]) -> dict[str, any] | None:
        _logger.debug(f"map: {item}")
        language_model = self._get_model()
        result = self._guidance_function(language_model, item, common=self._common_data)
        _logger.debug(f"Checking keys")
        for k in result.keys():
            assert k not in item, f"Duplicate key: {k}"

        _logger.debug(f"Updating item")
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
    processor = GuidanceAzureML(
        program_path=args.guidance_program,
        endpoint=args.azure_openai_endpoint,
        model=args.azure_openai_deployed_model,
        common_data=common_data,
    )

    # Run the processing
    line_map_mp(
        mapper=processor,
        source_file=args.input_dataset,
        dest_file=args.output_dataset,
        source_encoding=args.input_encoding,
        dest_encoding=args.output_encoding,
        error_file=args.error_dataset,
        error_encoding=args.error_encoding,
        n_worker_tasks=args.guidance_workers,
        max_errors=args.max_errors,
    )

    _logger.info("Complete")


if __name__ == "__main__":
    main()
