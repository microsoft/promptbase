import argparse
import functools
import json
import pathlib

from typing import Any

import mlflow

from aether_utils.jsonl_utils import line_reduce
from aether_utils.logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)


class Scorer:
    def __init__(self, correct_key: str, response_key: str):
        self.y_true = []
        self.y_pred = []
        self.dataset = []
        self.subject = []
        self.correct_key = correct_key
        self.response_key = response_key

    def __call__(self, line: dict[str, Any]):
        correct_answer = line[self.correct_key]
        response_answer = line[self.response_key]
        self.y_true.append(correct_answer)
        self.y_pred.append(response_answer)

    def generate_summary(self) -> dict[str, Any]:
        result = dict()

        result["n_answers"] = len(self.y_true)
        n_correct = 0
        for y_t, y_p in zip(self.y_true, self.y_pred):
            if y_t == y_p:
                n_correct += 1
        result["n_correct"] = n_correct
        result["accuracy"] = 0
        if len(self.y_true) > 0:
            result["accuracy"] = float(n_correct) / len(self.y_true)

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
    summary = scorer.generate_summary()

    _logger.info("Logging with mlflow")
    mlflow.log_metrics(summary)

    _logger.info("Writing output file")
    with open(args.output_dataset, encoding=args.output_encoding, mode="w") as jf:
        json.dump(summary, jf, indent=4)


if __name__ == "__main__":
    main()
