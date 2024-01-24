import argparse
import pathlib

from typing import Any

import datasets

from aether_utils.jsonl_file_utils import JSONLWriter
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

DATASET_OPTIONS = [*MMLU_DATASETS, "all_medicine_datasets", "all_mmlu_datasets"]

SPLITS = ["test", "validation", "dev"]


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)

    # Information about the ports
    ports_group = parser.add_argument_group("Ports")
    ports_group.add_argument("--output_dataset", type=pathlib.Path, required=True)
    ports_group.add_argument("--output_encoding", type=str, required=True)

    parser.add_argument(
        "--mmlu_dataset", type=str, choices=DATASET_OPTIONS, required=True
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
    _logger.info(f"Fetching {args.mmlu_dataset}")

    if args.mmlu_dataset == "all_medicine_datasets":
        target_datasets = [
            "anatomy",
            "clinical_knowledge",
            "college_biology",
            "college_medicine",
            "medical_genetics",
            "professional_medicine",
        ]
    elif args.mmlu_dataset == "all_mmlu_datasets":
        target_datasets = MMLU_DATASETS
    else:
        target_datasets = [args.mmlu_dataset]

    jsonl_writers: dict[str, JSONLWriter] = dict()
    for split in SPLITS:
        nxt_writer = JSONLWriter(
            args.output_dataset / f"{split}.jsonl", args.output_encoding
        )
        nxt_writer.__enter__()
        jsonl_writers[split] = nxt_writer

    for nxt_ds in target_datasets:
        _logger.info(f"Processing dataset {nxt_ds}")
        # Note that tasksource skips the huge 'train' file
        hf_data = datasets.load_dataset("tasksource/mmlu", nxt_ds)

        for split in SPLITS:
            _logger.info(f"Extracting split {split}")
            extracted_data = process_data_split(hf_data[split], subject=nxt_ds)
            _logger.info(f"Saving split {split}")
            for line in extracted_data:
                jsonl_writers[split].write_line(line)

    _logger.info("Closing JSONL files")
    for v in jsonl_writers.values():
        v.__exit__()

    _logger.info("Complete")


if __name__ == "__main__":
    main()
