import argparse
import functools
import json
import pathlib

from typing import Any

import mlflow
import sklearn.metrics as skm

from aether_utils.jsonl_utils import line_reduce
from aether_utils.logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)


class Scorer:
    def __init__(self, response_key: str):
        self.total_count = 0
        self.good_json_count = 0
        self.correct_name_count = 0
        self.correct_occupation_count = 0
        self.response_key = response_key

    def __call__(self, line: dict[str, Any]):
        self.total_count += 1
        response_answer = line[self.response_key]
        try:
            decoded_response = json.loads(response_answer)
            self.good_json_count += 1
            if decoded_response["name"] == line["entity"]:
                self.correct_name_count += 1
            if decoded_response["occupation"] == line["target_mediated"]:
                self.correct_occupation_count += 1
        except:
            pass

    def generate_summary(self) -> dict[str, Any]:
        result = dict()
        result["metrics"] = dict()

        result["metrics"]["total"] = self.total_count
        result["metrics"]["good_json"] = self.good_json_count
        result["metrics"]["correct_name"] = self.correct_name_count
        result["metrics"]["correct_occupation"] = self.correct_occupation_count
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
    mlflow.log_metrics(summary["metrics"].overall.to_dict())


if __name__ == "__main__":
    main()
