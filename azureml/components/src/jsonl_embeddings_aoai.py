import argparse
import functools
import pathlib

from urllib.parse import urlparse, parse_qs


from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from openai import AzureOpenAI

from shared.jsonl_utils_multiprocessing import line_map_mp
from shared.logging_utils import get_standard_logger_for_file


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


def get_aoai_client(
    endpoint: str,
) -> AzureOpenAI:
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    parsed_url = urlparse(endpoint)
    parsed_query = parse_qs(parsed_url.query)

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=parsed_query["api-version"],
    )
    return client


def process_item(
    item: dict[str, any],
    src_key: str,
    dst_key: str,
    azure_aoai_endpoint: str,
) -> dict[str, any]:
    _logger.info(f"process_item: {item}")

    client = get_aoai_client(azure_aoai_endpoint)

    parsed_url = urlparse(azure_aoai_endpoint)
    deployment_name = parsed_url.path.split("/")[3]
    _logger.info(f"Got Deployment: {deployment_name}")

    embeddings = (
        client.embeddings.create(input=[item[src_key]], model=deployment_name)
        .data[0]
        .embedding
    )

    _logger.info(f"Updating item")
    item[dst_key] = embeddings

    return item


def main():
    args = parse_args()

    # Bind arguments to the processor function
    processor = functools.partial(
        process_item,
        src_key=args.source_key,
        dst_key=args.destination_key,
        azure_aoai_endpoint=args.azure_openai_endpoint,
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
        n_worker_tasks=args.workers,
        n_errors_max=args.max_errors,
    )

    _logger.info("Complete")


if __name__ == "__main__":
    main()
