import json
import logging


from azure.ai.ml import dsl, Input
from azure.ai.ml.entities import Pipeline

from azureml_utils import ComponentCollector
from configs import AOAIConfig

_logger = logging.getLogger(__file__)
_logger.setLevel(logging.INFO)


def create_knn_fewshot_pipeline(
    *,
    components: ComponentCollector,
    embedding_config: AOAIConfig,
    inference_config: AOAIConfig,
    input_dataset: Input,
    example_dataset: Input,
    guidance_program: Input,
    num_examples: int,
) -> Pipeline:
    _logger.info(f"Starting create_knn_pipeline")

    question_key = "question"
    embedding_key = "embedding"
    fewshot_examples_key = "fewshot_examples"

    @dsl.pipeline(
        name=f"knn_pipeline",
        display_name=f"Answer with kNN Fewshots",
    )
    def knn_fewshot(guidance_prog: Input, input_ds: Input, example_ds: Input):
        embedding_outputs = dict()
        for k, v in dict(input=input_ds, example=example_ds).items():
            embedding_job = components.jsonl_embeddings(
                input_dataset=v,
                source_key=question_key,
                destination_key=embedding_key,
                workers=embedding_config.workers,
                max_errors=embedding_config.max_errors,
                azure_openai_endpoint=embedding_config.endpoint,
            )
            embedding_job.compute = embedding_config.compute_target
            embedding_job.name = f"add_embeddings_{k}"
            embedding_outputs[k] = embedding_job.outputs.output_dataset

        knn_job = components.jsonl_knn_cosine_similarity(
            input_dataset=embedding_outputs["input"],
            example_dataset=embedding_outputs["example"],
            input_vector_key=embedding_key,
            example_vector_key=embedding_key,
            output_key=fewshot_examples_key,
            k_nearest=num_examples,
        )
        knn_job.name = f"select_knn_cosine_similarity"

        guidance_job = components.jsonl_guidance(
            guidance_program=guidance_prog,
            guidance_workers=inference_config.workers,
            max_errors=inference_config.max_errors,
            input_dataset=knn_job.outputs.output_dataset,
            azure_openai_endpoint=inference_config.endpoint,
            azure_openai_deployed_model=inference_config.model,
        )
        guidance_job.name = f"guidance_fewshot"
        guidance_job.compute = inference_config.compute_target

        filter_job = components.jsonl_key_filter(
            input_dataset=guidance_job.outputs.output_dataset,
            drop_keys=json.dumps([fewshot_examples_key]),
        )
        filter_job.name = f"remove_intermediate_keys"

        return {"output_dataset": filter_job.outputs.output_dataset}

    sub_pipeline = knn_fewshot(guidance_program, input_dataset, example_dataset)

    return sub_pipeline
