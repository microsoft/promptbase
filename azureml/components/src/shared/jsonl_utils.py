# Copied from Medprompt.... perhaps those utils should go to PyPi?

import json
import pathlib
import tempfile

from typing import Any, Callable, Dict, List, Optional, Tuple

from .logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)


def load_jsonl(file_path: pathlib.Path, source_encoding: str) -> List[Dict[str, Any]]:
    result = []
    _logger.info(f"Loading JSON file: {file_path}")
    with open(file_path, "r", encoding=source_encoding) as jlf:
        current_line = 0
        for l in jlf:
            _logger.info(f"Processing line: {current_line}")
            nxt = json.loads(l)
            result.append(nxt)
            current_line += 1
    return result


def save_jsonl(
    file_path: pathlib.Path, data: List[Dict[str, Any]], destination_encoding: str
):
    _logger.info(f"Saving file {file_path}")
    with open(file_path, "w", encoding=destination_encoding) as out_file:
        for i, d in enumerate(data):
            _logger.info(f"Writing element {i}")
            d_str = json.dumps(d)
            out_file.write(d_str)
            out_file.write("\n")


def line_map(
    *,
    map_func: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    source_file: pathlib.Path,
    dest_file: pathlib.Path,
    source_encoding: str,
    dest_encoding: str,
    error_file: Optional[pathlib.Path] = None,
    error_encoding: Optional[str] = None,
    max_errors: int = -1,
) -> Tuple[int, int]:
    """Iterate over a JSONL file, applying map_func to each line"""
    assert source_file.exists()

    # If error_file is not specified, set up a temporary file
    def get_error_file(error_file_path: Optional[pathlib.Path]):
        if error_file_path:
            return open(error_file_path, "a", encoding=error_encoding)
        else:
            return tempfile.TemporaryFile(mode="w", encoding="utf-8-sig")

    successful_lines = 0
    error_lines = 0
    with open(source_file, "r", encoding=source_encoding) as in_file:
        with open(dest_file, "w", encoding=dest_encoding) as out_file:
            with get_error_file(error_file) as err_file:
                current_line = 0
                for nxt in in_file:
                    _logger.info(f"Processing line: {current_line}")
                    nxt_dict = json.loads(nxt)
                    try:
                        nxt_output = map_func(nxt_dict)
                        if nxt_output is not None:
                            nxt_output_string = json.dumps(nxt_output)
                            _logger.info(f"Writing output: {nxt_output_string}")
                            out_file.write(nxt_output_string)
                            out_file.write("\n")
                        else:
                            _logger.info(f"Skipping because map_func returned 'None'")
                        successful_lines += 1
                    except Exception as e:
                        _logger.warn(f"Caught exception: {e}")
                        err_file.write(nxt)
                        error_lines += 1
                    current_line += 1

                    if max_errors > 0 and error_lines > max_errors:
                        raise ValueError(f"Terminating after {error_lines} errors")
    _logger.info(
        f"line_map complete ({successful_lines} successes, {error_lines} failures)"
    )
    return successful_lines, error_lines


def line_reduce(
    *,
    reducer: Callable[[Dict[str, Any]], None],
    source_file: pathlib.Path,
    source_encoding: str,
):
    assert source_file.exists()

    with open(source_file, "r", encoding=source_encoding) as in_file:
        current_line = 0
        for nxt in in_file:
            _logger.info(f"Processing line: {current_line}")
            nxt_dict = json.loads(nxt)
            _logger.info(f"Calling reducer")
            reducer(nxt_dict)
    _logger.info(f"line_reduce complete")
