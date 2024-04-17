import argparse
import pathlib
import random

from typing import Any, Dict, List

from aether_utils.jsonl_file_utils import load_jsonl, save_jsonl
from aether_utils.logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)

    # Information about the datasets
    datasets_group = parser.add_argument_group("Datasets")
    datasets_group.add_argument("--input_dataset", type=pathlib.Path, required=True)
    datasets_group.add_argument("--input_encoding", type=str, required=True)
    datasets_group.add_argument("--output_dataset", type=pathlib.Path, required=True)
    datasets_group.add_argument("--output_encoding", type=str, required=True)

    # Information about the sampling
    sampling_group = parser.add_argument_group("Sampling")
    sampling_group.add_argument("--n_samples", type=int, required=True)
    sampling_group.add_argument("--random_seed", type=int, required=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    _logger.info("Loading input")
    all_data = load_jsonl(args.input_dataset, args.input_encoding)
    _logger.info(f"Loaded {len(all_data)} items")

    random.seed(args.random_seed)
    sampled_data = random.sample(all_data, k=args.n_samples)

    _logger.info("Saving output")
    save_jsonl(
        file_path=args.output_dataset,
        data=sampled_data,
        destination_encoding=args.output_encoding,
    )
    _logger.info("Done")


if __name__ == "__main__":
    main()
