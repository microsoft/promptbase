import argparse
import pathlib
import time

from azure.identity import DefaultAzureCredential

from azure.ai.ml import Input, MLClient, load_component, load_environment

from azure.ai.ml.entities import Component, Environment

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

    parser.add_argument(
        "--guidance_program",
        type=pathlib.Path,
        required=True,
        help="Path to the guidance program to be run",
    )

    args = parser.parse_args()
    return args


def create_environment_from_yaml(
    ml_client: MLClient, yaml_path: pathlib.Path, version_string: str
) -> Environment:
    _logger.info(f"Loading {yaml_path}")
    loaded_yaml = load_environment(source=yaml_path)
    _logger.info("Changing version")
    loaded_yaml.version = version_string
    _logger.info("Creating Environment")
    my_env = ml_client.environments.create_or_update(loaded_yaml)
    _logger.info(f"Environment {my_env.name}:{my_env.version} created")
    return my_env


def create_component_from_yaml(
    ml_client: MLClient,
    yaml_path: pathlib.Path,
    version_string: str,
    environment: Environment = None,
) -> Component:
    _logger.info(f"Loading {yaml_path}")
    loaded_yaml = load_component(source=yaml_path)
    _logger.info("Changing version")
    loaded_yaml.version = version_string
    _logger.info("Changing environment")
    loaded_yaml.environment = environment
    _logger.info("Creating component")
    my_comp = ml_client.components.create_or_update(loaded_yaml)
    _logger.info(f"Component {my_comp.name}:{my_comp.version} created")
    return my_comp


def main():
    args = parse_args()
    assert args.workspace_config.exists(), f"Could not find {args.workspace_config}"
    assert args.guidance_program.exists(), f"Could not find {args.guidance_program}"

    version_string = str(int(time.time()))
    _logger.info(f"AzureML object version for this run: {version_string}")

    _logger.info("Creating AzureML client")
    credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
    ml_client = MLClient.from_config(credential, path=args.workspace_config)

    _logger.info("Obtaining MMLU dataset")
    data = ml_client.data.get(name=args.dataset_name, label="latest")

    _logger.info("Creating the Promptbase Environment")
    promptbase_env = create_environment_from_yaml(
        ml_client,
        pathlib.Path("./environments/promptbase-basic-env.yaml"),
        version_string,
    )

    _logger.info("Creating the Guidance AOAI component")
    jsonl_guidance_aoai = create_component_from_yaml(
        ml_client,
        pathlib.Path("./components/jsonl_guidance_component.yaml"),
        version_string=version_string,
        environment=promptbase_env,
    )

    _logger.info("Registering the guidance program as an Input")
    guidance_program_ds = Input(
        type="uri_file",
        path=args.guidance_program,
        model="download",
    )

    _logger.info("Script Complete. Monitor experiment in AzureML Portal")


if __name__ == "__main__":
    main()
