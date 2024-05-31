import argparse
import json
import pathlib
import time

from azure.identity import DefaultAzureCredential

from azure.ai.ml import dsl, Input, MLClient, load_component, load_environment

from azure.ai.ml.entities import Component, Environment, Pipeline

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

    parser.add_argument(
        "--other_config",
        type=pathlib.Path,
        required=True,
        help="Path to file containing other configuration information",
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
    assert args.other_config.exists(), f"Couldnot find {args.other_config}"

    with open(args.other_config, "r") as jf:
        other_config = json.load(jf)
    _logger.info(f"Read in {args.other_config}")

    version_string = str(int(time.time()))
    _logger.info(f"AzureML object version for this run: {version_string}")

    _logger.info("Creating AzureML client")
    credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
    ml_client = MLClient.from_config(credential, path=args.workspace_config)

    _logger.info("Obtaining MMLU dataset")
    mmlu_ds = ml_client.data.get(name=args.dataset_name, label="latest")

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

    _logger.info("Creating the scorer component")
    jsonl_score_multiplechoice = create_component_from_yaml(
        ml_client,
        pathlib.Path("./components/jsonl_score_multiplechoice_component.yaml"),
        version_string=version_string,
        environment=promptbase_env,
    )

    _logger.info("Registering the guidance program as an Input")
    guidance_program_ds = Input(
        type="uri_file",
        path=args.guidance_program,
        model="download",
    )

    # -------------------------------------

    @dsl.pipeline()
    def basic_pipeline(guidance_program_input, dataset_input) -> Pipeline:
        guidance_job = jsonl_guidance_aoai(
            guidance_program=guidance_program_input,
            guidance_workers=4,
            max_errors=10,
            input_dataset=dataset_input,
            azure_openai_endpoint=other_config["aoai_endpoint"],
            azure_openai_deployment=other_config["aoai_deployment"],
            azure_openai_model=other_config["aoai_model"],
            azure_openai_api_version=other_config["aoai_api_version"],
        )
        guidance_job.name = "run_aoai_guidance"
        guidance_job.compute = other_config["aoai_compute"]

        score_job = jsonl_score_multiplechoice(
            input_dataset=guidance_job.outputs.output_dataset,
            correct_key="correct_answer",
            response_key="zero_shot_choice",
        )
        score_job.name = "score_results"

    constructed_pipeline = basic_pipeline(guidance_program_ds, mmlu_ds)
    constructed_pipeline.display_name = None
    constructed_pipeline.experiment_name = f"simple_{args.dataset_name}"
    constructed_pipeline.compute = other_config["general_compute"]

    _logger.info("Submitting pipeline")
    submitted_job = ml_client.jobs.create_or_update(constructed_pipeline)
    _logger.info(f"Submitted: {submitted_job.name}")

    _logger.info("Script Complete. Monitor experiment in AzureML Portal")


if __name__ == "__main__":
    main()
