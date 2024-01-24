import argparse
import functools
import pathlib

from aether_utils.jsonl_utils import line_map
from aether_utils.logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)

    # Information about the ports
    ports_group = parser.add_argument_group("Ports")
    ports_group.add_argument("--input_dataset", type=pathlib.Path, required=True)
    ports_group.add_argument("--input_encoding", type=str, required=True)
    ports_group.add_argument("--output_dataset", type=pathlib.Path, required=True)
    ports_group.add_argument("--output_encoding", type=str, required=True)

    # Information about the keys
    keys_group = parser.add_argument_group("Keys")
    keys_group.add_argument("--correct_key", type=str, required=True)
    keys_group.add_argument("--response_key", type=str, required=True)

    args = parser.parse_args()

    return args


def process_item(
    item: dict[str, any], *, correct_key: str, response_key: str
) -> dict[str, any]:
    result = None
    if item[correct_key] == item[response_key]:
        result = item
    return result


def main():
    args = parse_args()

    processor = functools.partial(
        process_item, correct_key=args.correct_key, response_key=args.response_key
    )

    s, f = line_map(
        map_func=processor,
        source_file=args.input_dataset,
        dest_file=args.output_dataset,
        source_encoding=args.input_encoding,
        dest_encoding=args.output_encoding,
    )
    _logger.info(f"Complete with {s} successes and {f} failures")


if __name__ == "__main__":
    main()
