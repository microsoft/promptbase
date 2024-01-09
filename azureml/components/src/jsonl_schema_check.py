import argparse
import functools
import json
import pathlib

from typing import Any, Dict, List

from jsonschema.protocols import Validator
from jsonschema.validators import Draft202012Validator

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
    datasets_group.add_argument("--schema_dataset", type=pathlib.Path, required=True)
    datasets_group.add_argument("--schema_encoding", type=str, required=True)
    datasets_group.add_argument("--output_dataset", type=pathlib.Path, required=True)
    datasets_group.add_argument("--output_encoding", type=str, required=True)
    datasets_group.add_argument("--error_dataset", type=pathlib.Path, required=True)
    datasets_group.add_argument("--error_encoding", type=str, required=True)

    # Forbidden keys
    parser.add_argument("--forbidden_keys", type=json_loads_fixer, required=True)

    # Maximum error count
    parser.add_argument("--max_errors", type=int, required=True)

    args = parser.parse_args()
    return args


def process_item(
    item: Dict[str, Any], *, json_validator: Validator, forbidden_keys=list[str]
) -> Dict[str, Any]:
    for k in forbidden_keys:
        assert k not in item, f"Key {k} not allowed"

    json_validator.validate(item)

    return item


def main():
    args = parse_args()

    # Load in the JSON schema
    with open(args.schema_dataset, "r", encoding=args.schema_encoding) as sf:
        json_schema = json.load(sf)

    # Check the schema
    Draft202012Validator.check_schema(json_schema)

    # Create the validator object
    validator = Draft202012Validator(schema=json_schema)

    processor = functools.partial(
        process_item, json_validator=validator, forbidden_keys=args.forbidden_keys
    )

    line_map(
        map_func=processor,
        source_file=args.input_dataset,
        dest_file=args.output_dataset,
        source_encoding=args.input_encoding,
        dest_encoding=args.output_encoding,
        error_file=args.error_dataset,
        error_encoding=args.error_encoding,
        max_errors=args.max_errors,
    )
    _logger.info("Complete")


if __name__ == "__main__":
    main()
