import argparse
import functools
import pathlib

import numpy as np


from aether_utils.jsonl_file_utils import load_jsonl
from aether_utils.jsonl_utils import line_map
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


def compute_knn(
    item: dict[str, any],
    *,
    examples: list[dict[str, any]],
    example_embedding_matrix: np.ndarray,
    input_vector_key: str,
    output_key: str,
    k_nearest: int,
) -> dict[str, any]:
    _logger.debug(f"process_item: {item}")

    item_embedding = np.asarray(item[input_vector_key])
    _logger.debug(f"Item embedding {item_embedding.dtype} {item_embedding.shape}")

    similarities = np.matmul(example_embedding_matrix, item_embedding)
    # np.argsort is ascending, so we need to reverse
    sorted_indices = list(reversed(np.argsort(similarities).tolist()))
    top_k_indices = sorted_indices[0:k_nearest]
    _logger.debug(f"k nearest: {top_k_indices}")
    k_examples = []
    for k in top_k_indices:
        k_examples.append(examples[k])
    item[output_key] = k_examples
    del item[input_vector_key]

    return item


def normalised_vector(input: list[float]) -> np.ndarray:
    result = np.asarray(input)
    result = result / np.linalg.norm(result)

    return result


def main():
    args = parse_args()

    example_data = load_jsonl(args.example_dataset, args.example_encoding)
    example_embedding_matrix = np.stack(
        [normalised_vector(e[args.example_vector_key]) for e in example_data], axis=0
    )
    _logger.info(
        f"Embedding Matrix: {example_embedding_matrix.dtype} {example_embedding_matrix.shape}"
    )

    # Remove the vectors
    for e in example_data:
        del e[args.example_vector_key]

    # Construct the mapping function
    processor = functools.partial(
        compute_knn,
        examples=example_data,
        example_embedding_matrix=example_embedding_matrix,
        input_vector_key=args.input_vector_key,
        output_key=args.output_key,
        k_nearest=args.k_nearest,
    )

    s, f = line_map(
        map_func=processor,
        source_file=args.input_dataset,
        source_encoding=args.input_encoding,
        dest_file=args.output_dataset,
        dest_encoding=args.output_encoding,
    )

    _logger.info(f"Complete with {s} successes and {f} failures")


if __name__ == "__main__":
    main()
