# Submit a run using:
# python .\submit_mmlu_zeroshot.py -cn zeroshot_config

import time

from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore

import omegaconf

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from azure.ai.ml import dsl, Input, MLClient
from azure.ai.ml.entities import Pipeline

from azureml_pipelines import create_zeroshot_pipeline
from azureml_utils import get_component_collector
from configs import AMLConfig, BiosBiasJSONPipelineConfig
from constants import GUIDANCE_PROGRAMS_DIR
from logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)


@dataclass
class PipelineConfig:
    zeroshot_config: BiosBiasJSONPipelineConfig = omegaconf.MISSING
    azureml_config: AMLConfig = omegaconf.MISSING


cs = ConfigStore.instance()
cs.store(name="config", node=PipelineConfig)


def create_biosbias_simple_json_pipeline(
    ml_client: MLClient, run_config: BiosBiasJSONPipelineConfig, version_string: str
):
    components = get_component_collector(ml_client, version_string)

    guidance_input = Input(
        type="uri_file",
        path=GUIDANCE_PROGRAMS_DIR / run_config.json_guidance_program,
        model="download",
    )

    ds_parts = run_config.biosbias_dataset.split(":")
    bios_ds = ml_client.data.get(ds_parts[0], version=ds_parts[1])

    inference_config = run_config.aoai_config

    @dsl.pipeline()
    def basic_pipeline() -> Pipeline:
        guidance_job = components.jsonl_guidance(
            guidance_program=guidance_input,
            guidance_workers=inference_config.workers,
            max_errors=inference_config.max_errors,
            input_dataset=bios_ds,
            azure_openai_endpoint=inference_config.endpoint,
            azure_openai_deployed_model=inference_config.model,
        )
        guidance_job.name = f"guidance_simple"
        guidance_job.compute = inference_config.compute_target

        score_job = components.jsonl_score_biosbias_json(
            input_dataset=guidance_job.outputs.output_dataset,
            response_key="model_answer",
        )
        score_job.name = f"score_biosbias_json"

    pipeline = basic_pipeline()
    pipeline.experiment_name = (
        f"{run_config.pipeline.base_experiment_name}_{ds_parts[0]}_{ds_parts[1]}"
    )
    pipeline.display_name = None
    pipeline.compute = run_config.pipeline.default_compute_target
    if run_config.pipeline.tags:
        pipeline.tags.update(run_config.tags)
    _logger.info("Pipeline created")

    return pipeline


@hydra.main(config_path="configs", version_base="1.1")
def main(config: PipelineConfig):
    version_string = str(int(time.time()))
    _logger.info(f"AzureML object version for this run: {version_string}")

    _logger.info(f"Azure Subscription: {config.azureml_config.subscription_id}")
    _logger.info(f"Resource Group: {config.azureml_config.resource_group}")
    _logger.info(f"Workspace : {config.azureml_config.workspace_name}")

    credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)

    ws_client = MLClient(
        credential=credential,
        subscription_id=config.azureml_config.subscription_id,
        resource_group_name=config.azureml_config.resource_group,
        workspace_name=config.azureml_config.workspace_name,
        logging_enable=False,
    )

    pipeline = create_biosbias_simple_json_pipeline(
        ws_client, config.zeroshot_config, version_string
    )
    _logger.info("Submitting pipeline")
    submitted_job = ws_client.jobs.create_or_update(pipeline)
    _logger.info(f"Submitted: {submitted_job.name}")


if __name__ == "__main__":
    main()
