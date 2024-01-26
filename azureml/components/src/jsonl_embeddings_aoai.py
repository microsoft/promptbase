import argparse
import pathlib

from urllib.parse import urlparse, parse_qs


from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from openai import AzureOpenAI

from aether_utils.jsonl_utils_multiprocessing import line_map_mp, ItemMapper
from aether_utils.logging_utils import get_standard_logger_for_file


_logger = get_standard_logger_for_file(__file__)


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

    # Processing configuration
    processing_group = parser.add_argument_group("Processing configuration")
    processing_group.add_argument("--workers", type=int, required=True)
    processing_group.add_argument("--max_errors", type=int, required=True)

    # Information about the embeddings mode
    model_group = parser.add_argument_group("Model Endpoint")
    model_group.add_argument("--azure_openai_endpoint", type=str, required=True)

    # Information about the keys
    keys_group = parser.add_argument_group("JSON Keys")
    keys_group.add_argument("--source_key", type=str, required=True)
    keys_group.add_argument("--destination_key", type=str, required=True)

    args = parser.parse_args()
    return args


class AOAIEmbedder(ItemMapper):
    def __init__(self, endpoint: str, src_key: str, dst_key: str):
        super().__init__()
        self._endpoint = endpoint
        self._src_key = src_key
        self._dst_key = dst_key

    def start_up(self, worker_id: int) -> None:
        _logger.info(f"Starting up {worker_id}")
        self._azure_credential = DefaultAzureCredential()

    def _get_aoai_client(self) -> AzureOpenAI:
        token_provider = get_bearer_token_provider(
            self._azure_credential, "https://cognitiveservices.azure.com/.default"
        )
        assert token_provider is not None

        # Pending a fix going into the released version of guidance,
        # we can only work with chat models
        parsed_url = urlparse(self._endpoint)
        parsed_query = parse_qs(parsed_url.query)

        client = AzureOpenAI(
            azure_endpoint=self._endpoint,
            azure_ad_token_provider=token_provider,
            api_version=parsed_query["api-version"],
        )
        return client

    def map(self, item: dict[str, any]) -> dict[str, any] | None:
        _logger.debug(f"map: {item}")

        client = self._get_aoai_client()

        parsed_url = urlparse(self._endpoint)
        deployment_name = parsed_url.path.split("/")[3]
        _logger.debug(f"Got Deployment: {deployment_name}")

        embeddings = (
            client.embeddings.create(input=[item[self._src_key]], model=deployment_name)
            .data[0]
            .embedding
        )

        _logger.debug(f"Updating item")
        item[self._dst_key] = embeddings

        return item


def main():
    args = parse_args()

    # Bind arguments to the processor function
    processor = AOAIEmbedder(
        src_key=args.source_key,
        dst_key=args.destination_key,
        endpoint=args.azure_openai_endpoint,
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
        n_worker_tasks=args.workers,
        max_errors=args.max_errors,
    )

    _logger.info("Complete")


if __name__ == "__main__":
    main()
