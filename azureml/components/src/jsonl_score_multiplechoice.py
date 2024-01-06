import argparse
import json
import pathlib

from typing import Any

import mlflow
import sklearn.metrics as skm

from shared.jsonl_utils import line_reduce
from shared.logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)


class Scorer:
    def __init__(self, correct_key: str, response_key: str):
        self.y_true = []
        self.y_pred = []
        self.correct_key = correct_key
        self.response_key = response_key

    def __call__(self, line: dict[str, Any]):
        correct_answer = line[self.correct_key]
        response_answer = line[self.response_key]
        self.y_true.append(correct_answer)
        self.y_pred.append(response_answer)

    def generate_summary(self) -> dict[str, Any]:
        result = dict()
        result["metrics"] = dict()
        result["metrics"]["n_total"] = len(self.y_true)
        result["metrics"]["accuracy"] = skm.accuracy_score(self.y_true, self.y_pred)
        result["metrics"]["n_correct"] = int(
            skm.accuracy_score(self.y_true, self.y_pred, normalize=False)
        )
        result["figures"] = dict()
        cm_display = skm.ConfusionMatrixDisplay.from_predictions(
            self.y_true, self.y_pred
        )
        result["figures"]["confusion_matrix"] = cm_display.figure_
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

    mlflow.log_metrics(summary["metrics"])
    for k, v in summary["figures"].items():
        mlflow.log_figure(v, f"{k}.png")

    _logger.info("Writing output file")
    with open(args.output_dataset, encoding=args.output_encoding, mode="w") as jf:
        json.dump(summary["metrics"], jf, indent=4)


if __name__ == "__main__":
    main()
