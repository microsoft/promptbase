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
from configs import AMLConfig, GSM8KZeroOrFewShotConfig
from constants import GUIDANCE_PROGRAMS_DIR
from logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)


@dataclass
class PipelineConfig:
    zeroorfewshot_config: GSM8KZeroOrFewShotConfig = omegaconf.MISSING
    azureml_config: AMLConfig = omegaconf.MISSING


cs = ConfigStore.instance()
cs.store(name="config", node=PipelineConfig)


def create_gsm8k_zeroshot_pipeline(
    ml_client: MLClient, run_config: GSM8KZeroOrFewShotConfig, version_string: str
):
    components = get_component_collector(ml_client, version_string)

    guidance_inputs = dict()
    for prog_filename in run_config.json_guidance_programs:
        k = prog_filename[0:-3]
        v = Input(
            type="uri_file",
            path=GUIDANCE_PROGRAMS_DIR / prog_filename,
            model="download",
        )
        guidance_inputs[k] = v
    _logger.info(f"Found {len(guidance_inputs)} guidance programs")

    @dsl.pipeline()
    def basic_pipeline() -> Pipeline:
        mmlu_fetch_job = components.jsonl_gsm8k_fetch()
        mmlu_fetch_job.name = f"fetch_gsm8k"

        split_outputs = dict()
        for s in ["train", "test"]:
            get_split_job = components.uri_folder_to_file(
                input_dataset=mmlu_fetch_job.outputs.output_dataset,
                filename_pattern=f"{s}.jsonl",
            )
            get_split_job.name = f"extract_split_{s}"
            split_outputs[s] = get_split_job.outputs.output_dataset

        sample_lines_job = components.jsonl_sample_lines(
            input_dataset=split_outputs["train"],
            n_samples=run_config.n_samples,
            random_seed=run_config.sample_random_seed
        )
        sample_lines_job.name= f"sample_{run_config.n_samples}_lines"

        random_examples_job = components.jsonl_random_examples(
            input_dataset=sample_lines_job.outputs.output_dataset,
            example_dataset=split_outputs["test"],
            output_key="examples",
            num_examples=run_config.n_fewshot,
            random_seed=run_config.fewshot_random_seed
        )
        random_examples_job.name=f"add_random_examples"

        for progname, prog_input in guidance_inputs.items():

            guidance_job = components.jsonl_guidance_mistral7b(
                guidance_program=prog_input,
                input_dataset=random_examples_job.outputs.output_dataset,
            )
            guidance_job.compute = run_config.llamacpp_config.compute_target
            guidance_job.name = f"guidance_mistral7b_{progname}"

            score_job = components.jsonl_score_numeric(
                input_dataset=guidance_job.outputs.output_dataset,
                correct_key="answer",
                response_key="zero_or_few_shot_answer",
            )
            score_job.name = f"score_{progname}"

    pipeline = basic_pipeline()
    pipeline.experiment_name = f"{run_config.pipeline.base_experiment_name}"
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

    pipeline = create_gsm8k_zeroshot_pipeline(
        ws_client, config.zeroorfewshot_config, version_string
    )
    _logger.info("Submitting pipeline")
    submitted_job = ws_client.jobs.create_or_update(pipeline)
    _logger.info(f"Submitted: {submitted_job.name}")


if __name__ == "__main__":
    main()
