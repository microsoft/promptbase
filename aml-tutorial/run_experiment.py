import argparse
import pathlib
import time

from azure.identity import DefaultAzureCredential

from azure.ai.ml import MLClient

from aether_utils.logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--workspace_config",
        type=pathlib.Path,
        default=pathlib.Path("./config.json"),
        help="Path to config.json downloaded from AzureML workspace",
    )

    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to process"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.workspace_config.exists(), f"Could not find {args.workspace_config}"

    version_string = str(int(time.time()))
    _logger.info(f"AzureML object version for this run: {version_string}")

    _logger.info("Creating AzureML client")
    credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
    ml_client = MLClient.from_config(credential, path=args.workspace_config)

    _logger.info("Obtaining MMLU dataset")
    data = ml_client.data.get(name=args.dataset_name)
    import json
    print(json.dumps(data._to_dict(), indent=4))
    
    _logger.info("Script Complete. Monitor experiment in AzureML Portal")


if __name__ == "__main__":
    main()
