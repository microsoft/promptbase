import argparse
import pathlib
import tempfile
import time

from typing import Any

import datasets

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data

from azure.identity import DefaultAzureCredential

from aether_utils.jsonl_file_utils import save_jsonl
from aether_utils.logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)

MMLU_DATASETS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

SPLITS = ["test", "validation", "dev"]


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)

    mmlu_group = parser.add_argument_group(
        "MMLU Information", description="Options pertaining to the data"
    )
    mmlu_group.add_argument(
        "--mmlu_dataset",
        type=str,
        choices=MMLU_DATASETS,
        required=True,
        help="The name of the desired MMLU dataset",
    )
    mmlu_group.add_argument(
        "--split",
        type=str,
        choices=SPLITS,
        default="validation",
        help="Which of the splits to use",
    )

    aml_group = parser.add_argument_group(
        "AzureML Information", description="Options pertaining to AzureML"
    )
    aml_group.add_argument(
        "--workspace_config",
        type=pathlib.Path,
        default=pathlib.Path("./config.json"),
        help="Path to config.json downloaded from AzureML workspace",
    )

    args = parser.parse_args()
    return args


def process_data_split(data, subject: str) -> list[dict[str, Any]]:
    all_questions = []
    for line in data:
        nxt = dict(
            dataset="mmlu",
            subject=subject,
            question=line["question"],
            choices=line["choices"],
            correct_answer=line["answer"],
        )
        all_questions.append(nxt)

    return all_questions


def main():
    args = parse_args()
    assert args.workspace_config.exists(), f"Could not find {args.workspace_config}"

    _logger.info("Creating AzureML client")
    credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
    ml_client = MLClient.from_config(credential, path=args.workspace_config)

    _logger.info(f"Fetching {args.mmlu_dataset}")
    hf_data = datasets.load_dataset("tasksource/mmlu", args.mmlu_dataset)

    _logger.info(f"Reformatting data")
    all_questions = process_data_split(hf_data[args.split], args.mmlu_dataset)

    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = pathlib.Path(temp_dir)

        dataset_name = f"mmlu_{args.mmlu_dataset}_{args.split}"

        out_file = out_dir / f"{dataset_name}.jsonl"
        save_jsonl(out_file, data=all_questions, destination_encoding="utf-8-sig")

        aml_data = Data(
            name=dataset_name,
            version=str(int(time.time())),
            description="Sample multiple choice dataset",
            path=out_file,
            type=AssetTypes.URI_FILE,
        )
        returned_data = ml_client.data.create_or_update(aml_data)
        _logger.info(
            f"Created dataset {returned_data.name} at version {returned_data.version}"
        )

    _logger.info("Complete")


if __name__ == "__main__":
    main()
