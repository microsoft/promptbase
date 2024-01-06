import argparse
import functools
import json
import pathlib

import numpy as np


from shared.jsonl_file_utils import load_jsonl
from shared.jsonl_utils import line_map
from shared.logging_utils import get_standard_logger_for_file


_logger = get_standard_logger_for_file(__file__)


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)

    # Information about the datasets
    datasets_group = parser.add_argument_group("Datasets")
    datasets_group.add_argument("--input_dataset", type=pathlib.Path, required=True)
    datasets_group.add_argument("--input_encoding", type=str, required=True)
    datasets_group.add_argument("--output_dataset", type=pathlib.Path, required=True)
    datasets_group.add_argument("--output_encoding", type=str, required=True)
    datasets_group.add_argument("--example_dataset", type=pathlib.Path, required=True)
    datasets_group.add_argument("--example_encoding", type=str, required=True)

    # Information about keys
    key_group = parser.add_argument_group("Keys")
    key_group.add_argument("--input_vector_key", type=str, required=True)
    key_group.add_argument("--example_vector_key", type=str, required=True)
    key_group.add_argument("--output_key", type=str, required=True)

    # Information about the algorithm
    algo_group = parser.add_argument_group("Algorithm")
    algo_group.add_argument("--k_nearest", type=int, required=True)

    args = parser.parse_args()
    return args


def normalised_vector(input: list[float]) -> np.ndarray:
    result = np.asarray(input)
    result = result / np.linalg.norm(result)

    return result


def main():
    args = parse_args()

    example_data = load_jsonl(args.example_dataset, args.example_encoding)
    example_embedding_matrix = np.stack(
        [normalised_vector(e[args.example_vector_key]) for e in example_data], axis=-1
    )
    _logger.info(
        f"Embedding Matrix: {example_embedding_matrix.dtype} {example_embedding_matrix.shape}"
    )

    _logger.info("Complete")


if __name__ == "__main__":
    main()
