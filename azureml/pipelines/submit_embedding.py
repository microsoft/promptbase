# python .\pipelines\submit_embedding.py -cn embedding_config

import time

from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore

import omegaconf

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from azure.ai.ml import dsl, MLClient
from azure.ai.ml.entities import Pipeline

from azureml_utils import get_component_collector
from configs import AMLConfig, EmbeddingConfig
from logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)


@dataclass
class PipelineConfig:
    embedding_config: EmbeddingConfig = omegaconf.MISSING
    azureml_config: AMLConfig = omegaconf.MISSING


cs = ConfigStore.instance()
cs.store(name="config", node=PipelineConfig)


def create_embedding_pipeline(
    ml_client: MLClient, run_config: EmbeddingConfig, version_string: str
):
    components = get_component_collector(ml_client, version_string)

    @dsl.pipeline()
    def basic_pipeline() -> Pipeline:
        mmlu_fetch_job = components.jsonl_mmlu_fetch(
            mmlu_dataset=run_config.mmlu_dataset
        )
        mmlu_fetch_job.name = f"fetch_mmlu_{run_config.mmlu_dataset}"

        get_split_job = components.uri_folder_to_file(
            input_dataset=mmlu_fetch_job.outputs.output_dataset,
            filename_pattern=f"{run_config.mmlu_split}.jsonl",
        )
        get_split_job.name = f"extract_split_{run_config.mmlu_split}"

        embedding_job = components.jsonl_embeddings(
            input_dataset=get_split_job.outputs.output_dataset,
            source_key=run_config.source_key,
            destination_key=run_config.destination_key,
            workers=run_config.workers,
            max_errors=run_config.max_errors,
            azure_openai_endpoint=run_config.aoai_embedding_config.endpoint,
        )
        embedding_job.compute = run_config.aoai_embedding_config.compute_target
        embedding_job.name = f"add_embeddings_{run_config.mmlu_split}"

    pipeline = basic_pipeline()
    pipeline.experiment_name = (
        f"{run_config.pipeline.base_experiment_name}_{run_config.mmlu_dataset}"
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

    pipeline = create_embedding_pipeline(
        ws_client, config.embedding_config, version_string
    )
    _logger.info("Submitting pipeline")
    submitted_job = ws_client.jobs.create_or_update(pipeline)
    _logger.info(f"Submitted: {submitted_job.name}")


if __name__ == "__main__":
    main()
