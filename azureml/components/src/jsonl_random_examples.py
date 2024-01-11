import argparse
import functools
import pathlib
import random


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
    key_group.add_argument("--output_key", type=str, required=True)

    # Information about the algorithm
    algo_group = parser.add_argument_group("Algorithm")
    algo_group.add_argument("--num_examples", type=int, required=True)
    algo_group.add_argument("--random_seed", type=int, required=True)

    args = parser.parse_args()
    return args


def select_examples(
    item: dict[str, any],
    *,
    examples: list[dict[str, any]],
    num_examples: int,
    output_key: str,
) -> dict[str, any]:
    # Note that random.samples() is _without_ replacement
    selected_examples = random.sample(examples, num_examples)
    item[output_key] = selected_examples
    return item


def main():
    args = parse_args()

    example_data = load_jsonl(args.example_dataset, args.example_encoding)
    _logger.info("Loaded example file")
    random.seed(args.random_seed)

    # Construct the mapping function
    processor = functools.partial(
        select_examples,
        examples=example_data,
        output_key=args.output_key,
        num_examples=args.num_examples,
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
