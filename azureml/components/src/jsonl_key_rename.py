import argparse
import functools
import pathlib

from typing import Any, Dict, List

from shared.argparse_utils import json_loads_fixer
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

    # Renaming config
    parser.add_argument("--rename_keys", type=json_loads_fixer, required=True)

    args = parser.parse_args()
    return args


def process_item(item: Dict[str, Any], *, rename: Dict[str, str]) -> Dict[str, Any]:
    result = dict()

    _logger.info("Processing renames")
    for k in item:
        if k in rename:
            result[rename[k]] = item[k]
        else:
            result[k] = item[k]
    return result


def main():
    args = parse_args()

    assert len(args.rename_keys) > 0, "Must rename at least one key!"

    processor = functools.partial(process_item, rename=args.rename_keys)
    line_map(
        map_func=processor,
        source_file=args.input_dataset,
        dest_file=args.output_dataset,
        source_encoding=args.input_encoding,
        dest_encoding=args.output_encoding,
    )


if __name__ == "__main__":
    main()
