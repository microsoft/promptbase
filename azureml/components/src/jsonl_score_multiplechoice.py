import argparse
import json
import pathlib

from typing import Any

from shared.jsonl_utils import line_reduce
from shared.logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)


class Scorer:
    def __init__(self, correct_key: str, response_key: str):
        self.n_correct = 0
        self.n_total = 0
        self.correct_key = correct_key
        self.response_key = response_key

    def __call__(self, line: dict[str, Any]):
        self.n_total += 1
        correct_answer = line[self.correct_key]
        response_answer = line[self.response_key]
        if correct_answer == response_answer:
            self.n_correct += 1

    def generate_summary(self) -> dict[str, Any]:
        result = dict()
        result["n_correct"] = self.n_correct
        result["n_total"] = self.n_total
        result["accuracy"] = self.n_correct / self.n_total

        return result


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


def main():
    args = parse_args()

    scorer = Scorer(correct_key=args.correct_key, response_key=args.response_key)
    line_reduce(
        reducer=scorer,
        source_file=args.input_dataset,
        source_encoding=args.input_encoding,
    )

    _logger.info("Writing output file")
    with open(args.output_dataset, encoding=args.output_encoding, mode="w") as jf:
        json.dump(scorer.generate_summary(), jf, indent=4)


if __name__ == "__main__":
    main()
