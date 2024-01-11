import argparse
import pathlib

from typing import Any

import datasets

from shared.jsonl_file_utils import save_jsonl, JSONLWriter
from shared.logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)

MMLU_DATASETS = [
    "anatomy",
    "astronomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]
SPLITS = ["test", "validation", "dev"]


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)

    # Information about the ports
    ports_group = parser.add_argument_group("Ports")
    ports_group.add_argument("--output_dataset", type=pathlib.Path, required=True)
    ports_group.add_argument("--output_encoding", type=str, required=True)

    parser.add_argument(
        "--mmlu_dataset", type=str, choices=MMLU_DATASETS, required=True
    )

    args = parser.parse_args()
    return args


def process_data_split(data) -> list[dict[str, Any]]:
    all_questions = []
    for line in data:
        nxt = dict(
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
    else:
        target_datasets = [args.mmlu_dataset]

    jsonl_writers: dict[str, JSONLWriter] = dict()
    for split in SPLITS:
        nxt_writer = JSONLWriter(
            args.output_dataset / f"{split.jsonl}", args.output_encoding
        )
        jsonl_writers[split] = nxt_writer

    for nxt_ds in target_datasets:
        _logger.info(f"Processing dataset {nxt_ds}")
        # Note that tasksource skips the huge 'train' file
        hf_data = datasets.load_dataset("tasksource/mmlu", nxt_ds)

        for split in SPLITS:
            _logger.info(f"Extracting split {split}")
            extracted_data = process_data_split(hf_data[split])
            _logger.info(f"Saving split {split}")
            for line in extracted_data:
                jsonl_writers[split].write_line(line)

    _logger.info("Closing JSONL files")
    for v in jsonl_writers.values():
        v.__exit__()

    _logger.info("Complete")


if __name__ == "__main__":
    main()
