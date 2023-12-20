# Submit a run using:
# python .\submit_mmlu_zeroshot.py -cn zeroshot_config

import time

from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from azure.ai.ml import dsl, Input, MLClient
from azure.ai.ml.entities import Pipeline

from azureml_utils import get_component_collector
from configs import AMLConfig, ZeroShotRunConfig
from logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)


@dataclass
class PipelineConfig:
    zeroshot_config: ZeroShotRunConfig
    aml_config: AMLConfig


cs = ConfigStore.instance()
cs.store(name="config", node=PipelineConfig)


def create_zeroshot_pipeline(
    ml_client: MLClient, run_config: ZeroShotRunConfig, version_string: str
):
    components = get_component_collector(ml_client, version_string)

    @dsl.pipeline()
    def basic_pipeline() -> Pipeline:
        mmlu_fetch_job = components.jsonl_mmlu_fetch(
            mmlu_dataset=run_config.mmlu_dataset
        )
        mmlu_fetch_job.name = f"fetch_mmlu_{run_config.mmlu_dataset}"

    pipeline = basic_pipeline()
    pipeline.experiment_name = (
        f"{run_config.base_experiment_name}_{run_config.mmlu_dataset}"
    )
    pipeline.display_name = None
    pipeline.compute = run_config.default_compute_target
    if run_config.tags:
        pipeline.tags.update(run_config.tags)
    _logger.info("Pipeline created")

    return pipeline


@hydra.main(config_path="configs", version_base="1.1")
def main(config: PipelineConfig):
    version_string = str(int(time.time()))
    _logger.info(f"AzureML object version for this run: {version_string}")

    _logger.info(f"Azure Subscription: {config.aml_config.subscription_id}")
    _logger.info(f"Resource Group: {config.aml_config.resource_group}")
    _logger.info(f"Workspace : {config.aml_config.workspace_name}")

    credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)

    ws_client = MLClient(
        credential=credential,
        subscription_id=config.aml_config.subscription_id,
        resource_group_name=config.aml_config.resource_group,
        workspace_name=config.aml_config.workspace_name,
        logging_enable=False,
    )

    pipeline = create_zeroshot_pipeline(
        ws_client, config.zeroshot_config, version_string
    )
    _logger.info("Submitting pipeline")
    submitted_job = ws_client.jobs.create_or_update(pipeline)
    _logger.info(f"Submitted: {submitted_job.name}")


if __name__ == "__main__":
    main()
