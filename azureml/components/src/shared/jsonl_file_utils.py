import json
import pathlib
import tempfile

from typing import Any

from .logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)


class JSONLReader:
    """Line-by-line iteration over a JSONL file

    Can be used in a 'with' statement, and then iterated over.
    The returned value is a decoded JSON object, rather than
    the line itself
    """

    def __init__(self, jsonl_file: pathlib.Path, encoding: str):
        self._file_path = jsonl_file
        self._encoding = encoding
        self._jf = None

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        nxt_line = next(self._jf)
        result = json.loads(nxt_line)
        return result

    def __enter__(self):
        self._jf = open(self._file_path, "r", encoding=self._encoding)
        return self

    def __exit__(self, *args):
        self._jf.close()


class JSONLWriter:
    def __init__(self, jsonl_file: pathlib.Path | None, encoding: str | None):
        self._file_path = jsonl_file
        self._encoding = encoding
        self._jf = None

    def __enter__(self):
        if self._file_path is not None:
            self._jf = open(self._file_path, "w", encoding=self._encoding)
        else:
            _logger.info(f"No target path specified, writing to TemporaryFile")
            self._jf = tempfile.TemporaryFile(mode="w", encoding="utf-8-sig")
        return self

    def __exit__(self, *args):
        self._jf.close()

    def write_line(self, target_object: dict[str, Any]):
        nxt_line = json.dumps(target_object)
        self._jf.write(nxt_line)
        self._jf.write("\n")


def load_jsonl(file_path: pathlib.Path, source_encoding: str) -> list[dict[str, Any]]:
    result = []
    _logger.info(f"Loading JSON file: {file_path}")
    with JSONLReader(file_path, source_encoding) as jlf:
        current_line = 0
        for l in jlf:
            _logger.info(f"Processing line: {current_line}")
            result.append(l)
            current_line += 1
    return result


def save_jsonl(
    file_path: pathlib.Path, data: list[dict[str, Any]], destination_encoding: str
):
    _logger.info(f"Saving file {file_path}")
    with JSONLWriter(file_path, destination_encoding) as out_file:
        for i, d in enumerate(data):
            _logger.info(f"Writing element {i}")
            out_file.write_line(d)
