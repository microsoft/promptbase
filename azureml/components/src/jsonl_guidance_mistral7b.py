import argparse
import importlib.util
import json
import pathlib
import time

from typing import Any, Callable, Dict

import guidance

from huggingface_hub import hf_hub_download

import mlflow

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

    args = parser.parse_args()
    return args


class LLMProcessor:
    def __init__(
        self,
        program_path,
        model: guidance.models.Model,
        common_data: dict[str, any] | None,
    ):
        self._program_path = program_path
        self._model = model
        self._guidance_function = self._get_guidance_function()
        self._common_data = common_data
        self._step = 0

    def __call__(self, item: Dict[str, Any]) -> dict[str, any]:
        _logger.debug(f"__call__: {item}")
        start = time.time()
        result = self._guidance_function(self._model, item, common=self._common_data)
        stop = time.time()
        mlflow.log_metric("time_taken", value=stop - start, step=self._step)
        _logger.debug(f"Checking keys")
        for k in result.keys():
            assert k not in item, f"Duplicate key: {k}"

        _logger.debug(f"Updating item")
        item.update(**result)
        self._step += 1

        return item

    def _get_guidance_function(
        self,
    ) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        _logger.debug("Importing guidance file")
        spec = importlib.util.spec_from_file_location(USER_MODULE, self._program_path)
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

    repo_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    filename = "mistral-7b-instruct-v0.2.Q8_0.gguf"
    downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename)

    guidance_model = guidance.models.LlamaCpp(
        downloaded_file, verbose=True, n_gpu_layers=-1, n_ctx=4096
    )
    # _logger.info(f"guidance_model.device: {guidance_model.engine.device}")

    processor = LLMProcessor(
        program_path=args.guidance_program,
        model=guidance_model,
        common_data=common_data,
    )

    _logger.info("Starting to process input")
    s, f = line_map(
        map_func=processor,
        source_file=args.input_dataset,
        dest_file=args.output_dataset,
        source_encoding=args.input_encoding,
        dest_encoding=args.output_encoding,
    )

    _logger.info(f"Complete with {s} successes and {f} failures")


if __name__ == "__main__":
    main()
