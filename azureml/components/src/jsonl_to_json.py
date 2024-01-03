import argparse
import json
import pathlib


from shared.jsonl_utils import line_reduce
from shared.logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)


class ContentAccumulator:
    def __init__(self):
        self.contents = []

    def __call__(self, line: dict[str, any]):
        self.contents.append(line)


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)

    # Information about the ports
    ports_group = parser.add_argument_group("Ports")
    ports_group.add_argument("--input_dataset", type=pathlib.Path, required=True)
    ports_group.add_argument("--input_encoding", type=str, required=True)
    ports_group.add_argument("--output_dataset", type=pathlib.Path, required=True)
    ports_group.add_argument("--output_encoding", type=str, required=True)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    _logger.info("Starting accumulation")
    acc = ContentAccumulator()
    line_reduce(
        reducer=acc,
        source_file=args.input_dataset,
        source_encoding=args.input_encoding,
    )
    _logger.info("All lines accumulated")

    with open(args.output_dataset, "w", encoding=args.output_encoding) as jf:
        json.dump(acc.contents, jf, indent=4)


if __name__ == "__main__":
    main()
