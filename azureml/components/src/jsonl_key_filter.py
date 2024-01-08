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

    # Filtering config
    filtering_group = parser.add_mutually_exclusive_group(required=True)
    filtering_group.add_argument(
        "--keep_keys",
        type=json_loads_fixer,
        default=[],
        help="JSON list of keys to keep",
    )
    filtering_group.add_argument(
        "--drop_keys",
        type=json_loads_fixer,
        default=[],
        help="JSON list of keys to drop",
    )

    args = parser.parse_args()
    return args


def process_item(
    item: Dict[str, Any], *, keep: List[str], drop: List[str]
) -> Dict[str, Any]:
    result = dict()

    if len(keep) > 0:
        _logger.info("Processing keeps")
        for k in keep:
            result[k] = item[k]
    elif len(drop) > 0:
        _logger.info("Processing drops")
        for k, v in item.items():
            assert k in item, f"Key {k} not in original!"
            if k not in drop:
                result[k] = v
    else:
        raise ValueError("Shouldn't get here")

    return result


def main():
    args = parse_args()

    # Exclusivity taken care of by add_mutually_exclusive_group
    assert (
        len(args.keep_keys) > 0 or len(args.drop_keys) > 0
    ), "Must either keep or drop something!"

    processor = functools.partial(
        process_item, keep=args.keep_keys, drop=args.drop_keys
    )

    line_map(
        map_func=processor,
        source_file=args.input_dataset,
        dest_file=args.output_dataset,
        source_encoding=args.input_encoding,
        dest_encoding=args.output_encoding,
    )


if __name__ == "__main__":
    main()
