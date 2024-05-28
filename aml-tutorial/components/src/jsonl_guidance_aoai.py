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
    datasets_group.add_argument("--output_dataset", type=pathlib.Path, required=True)
    datasets_group.add_argument("--error_dataset", type=pathlib.Path, required=True)

    # Information about the guidance program
    parser.add_argument("--guidance_program", type=pathlib.Path, required=True)
    parser.add_argument("--guidance_workers", type=int, required=True)
    parser.add_argument("--max_errors", type=int, required=True)

    # Information about the model
    model_group = parser.add_argument_group("Model Endpoint")
    model_group.add_argument("--azure_openai_endpoint", type=str, required=True)
    model_group.add_argument("--azure_openai_deployment", type=str, required=True)
    model_group.add_argument("--azure_openai_model", type=str, required=True)
    model_group.add_argument("--azure_openai_api_version", type=str, required=True)

    args = parser.parse_args()
    return args


def get_guidance_function(
    program_path: pathlib.Path,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    _logger.info(f"Importing guidance file: {program_path}")
    spec = importlib.util.spec_from_file_location(USER_MODULE, program_path)
    module_definition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_definition)

    guidance_func = getattr(module_definition, GUIDANCE_FUNCTION)
    _logger.info("Guidance program imported")

    return guidance_func


class GuidanceAzureML(ItemMapper):
    def __init__(
        self,
        *,
        program_path: pathlib.Path,
        endpoint: str,
        deployment: str,
        model: str,
        api_version: str,
    ):
        super().__init__()
        self._program_path = program_path
        self._endpoint = endpoint
        self._deployment = deployment
        self._model = model
        self._api_version = api_version

    def start_up(self, worker_id: int) -> None:
        _logger.info(f"Starting up {worker_id}")
        self._guidance_function = get_guidance_function(self._program_path)
        self._azure_credential = DefaultAzureCredential()
        _logger.info(f"Start up complete {worker_id}")

    def _get_model(self) -> guidance.models.Model:
        token_provider = get_bearer_token_provider(
            self._azure_credential, "https://cognitiveservices.azure.com/.default"
        )
        assert token_provider is not None
        _logger.info(f"Got token_provider")

        azureai_model = guidance.models.AzureOpenAI(
            model=self._model,
            azure_endpoint=self._endpoint,
            azure_deployment=self._deployment,
            version=self._api_version,
            azure_ad_token_provider=token_provider,
        )
        _logger.info(f"Created AzureOpenAI model")

        return azureai_model

    def map(self, item: dict[str, any]) -> dict[str, any] | None:
        _logger.info(f"map: {item}")
        language_model = self._get_model()
        result = self._guidance_function(language_model, item)
        _logger.debug(f"Checking keys")
        for k in result.keys():
            assert k not in item, f"Duplicate key: {k}"

        _logger.info(f"Updating item")
        item.update(**result)

        return item


def main():
    args = parse_args()

    # Bind arguments to the processor function
    processor = GuidanceAzureML(
        program_path=args.guidance_program,
        endpoint=args.azure_openai_endpoint,
        deployment=args.azure_openai_deployment,
        model=args.azure_openai_model,
        api_version=args.azure_openai_api_version,
    )

    # Run the processing
    line_map_mp(
        mapper=processor,
        source_file=args.input_dataset,
        dest_file=args.output_dataset,
        source_encoding="utf-8-sig",
        dest_encoding="utf-8-sig",
        error_file=args.error_dataset,
        error_encoding="utf-8-sig",
        n_worker_tasks=args.guidance_workers,
        max_errors=args.max_errors,
    )

    _logger.info("Complete")


if __name__ == "__main__":
    main()
