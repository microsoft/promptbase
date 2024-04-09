import argparse
import json
import locale
import pathlib
import re

from typing import Any, Dict

import requests


from aether_utils.jsonl_file_utils import JSONLWriter, JSONLReader
from aether_utils.logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)

BASE_DATA_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/"

SPLITS = ["train", "test"]


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)

    # Information about the ports
    ports_group = parser.add_argument_group("Ports")
    ports_group.add_argument("--output_dataset", type=pathlib.Path, required=True)
    ports_group.add_argument("--output_encoding", type=str, required=True)

    args = parser.parse_args()
    return args


def extract_thought_parts(thought: str) -> Dict[str, Any]:
    thought_re = r"(.*)<<(.*=.*)>>(.*)"
    match = re.match(thought_re, thought)

    result = dict()
    if match:
        result["step"] = match.group(1)
        result["calculation"] = match.group(2)
        result["result"] = match.group(3)
    else:
        result["step"] = thought
    return result


def process_line(item: Dict[str, Any]) -> Dict[str, Any]:
    result = dict()
    _logger.debug(f"Processing {item}")

    result["question"] = item["question"]

    split_answer = item["answer"].split("####")
    result["answer"] = locale.atof(split_answer[1])

    result["thoughts"] = []
    for thought in split_answer[0].splitlines():
        result["thoughts"].append(extract_thought_parts(thought))
    return result


def main():
    args = parse_args()

    # For parsing numbers
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

    for split in SPLITS:
        _logger.info(f"Starting split {split}")
        target_url = f"{BASE_DATA_URL}{split}.jsonl"

        _logger.info(f"Fetching {target_url}")
        response = requests.get(target_url)
        assert response.status_code == 200, f"Got response {response}"

        with JSONLWriter(
            args.output_dataset / f"{split}.jsonl", args.output_encoding
        ) as jlw:
            for line in response.text.splitlines():
                nxt_item = json.loads(line)
                output_item = process_line(nxt_item)
                jlw.write_line(output_item)
        _logger.info(f"Completed split {split}")

    _logger.info("Complete")


if __name__ == "__main__":
    main()
