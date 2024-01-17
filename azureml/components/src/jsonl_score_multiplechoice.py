import argparse
import functools
import json
import pathlib

from typing import Any

import fairlearn.metrics as flm
import mlflow
import sklearn.metrics as skm

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
        if "dataset" in line:
            self.dataset.append(line["dataset"])
        else:
            self.dataset.append("No dataset")
        if "subject" in line:
            self.subject.append(line["subject"])
        else:
            self.subject.append("No subject")

    def generate_summary(self) -> dict[str, Any]:
        metrics = {
            "count": flm.count,
            "accuracy": skm.accuracy_score,
            "n_correct": functools.partial(skm.accuracy_score, normalize=False),
        }

        mf = flm.MetricFrame(
            metrics=metrics,
            y_true=self.y_true,
            y_pred=self.y_pred,
            sensitive_features=dict(dataset=self.dataset, subject=self.subject),
        )

        result = dict()
        result["metrics"] = mf
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

    _logger.info("Logging with mlflow")
    mlflow.log_metrics(summary["metrics"].overall.to_dict())
    for k, v in summary["figures"].items():
        mlflow.log_figure(v, f"{k}.png")

    _logger.info("Writing output file")

    by_group_dict = dict()
    # Due to how MetricFrame does its indexing, we have to unpack the
    # key into another level of nesting
    for k, v in summary["metrics"].by_group.to_dict(orient="index").items():
        if k[0] not in by_group_dict:
            by_group_dict[k[0]] = dict()
        by_group_dict[k[0]][k[1]] = v

    output_dict = dict(
        overall=summary["metrics"].overall.to_dict(),
        details=by_group_dict,
    )
    print(f"output_dict:\n {json.dumps(output_dict,indent=4)}")
    with open(args.output_dataset, encoding=args.output_encoding, mode="w") as jf:
        json.dump(output_dict, jf, indent=4)


if __name__ == "__main__":
    main()
