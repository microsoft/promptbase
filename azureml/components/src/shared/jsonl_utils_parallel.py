import dataclasses
import functools
import json
import pathlib
import tempfile
import time

from enum import StrEnum
from typing import Any, Callable

from shared.logging_utils import get_standard_logger_for_file

import joblib

_logger = get_standard_logger_for_file(__file__)


class JSONLFile:
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


class ItemState(StrEnum):
    Success = "Success"
    Failure = "Failure"


@dataclasses.dataclass
class MapResult:
    state: ItemState = ItemState.Success
    time: float = float()
    result: dict[str, Any] | None = None


def _map_wrapper(
    item: dict[str, Any],
    *,
    map_func: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> MapResult:
    _logger.info(f"Mapping :{item}")
    start = time.time()
    result = MapResult()
    try:
        result.result = map_func(item)
        result.state = ItemState.Success
    except Exception as e:
        _logger.warning(f"Caught exception: {e}")
        result.result = item
        result.state = ItemState.Failure
    stop = time.time()
    result.time = stop - start
    return result


def line_map_parallel(
    *,
    map_func: Callable[[dict[str, Any]], dict[str, Any] | None],
    source_file: pathlib.Path,
    source_encoding: str,
    dest_file: pathlib.Path,
    dest_encoding: str,
    n_worker_tasks: int,
    error_file: pathlib.Path | None = None,
    error_encoding: str | None = None,
    n_errors_max: int = 5,
):
    def get_error_file(error_file_path: pathlib.Path | None):
        if error_file_path:
            return open(error_file_path, "a", encoding=error_encoding)
        else:
            return tempfile.TemporaryFile(mode="w", encoding="utf-8-sig")

    actual_map_func = functools.partial(_map_wrapper, map_func=map_func)
    with JSONLFile(source_file, source_encoding) as jsonl_src:
        with open(dest_file, "w", encoding=dest_encoding) as out_file:
            result = joblib.Parallel(n_jobs=n_worker_tasks, return_as="generator")(
                joblib.delayed(actual_map_func)(x) for x in jsonl_src
            )
            for r in result:
                assert isinstance(r, MapResult)
                if r.state == ItemState.Success:
                    nxt_output = json.dumps(r.result)
                    out_file.write(nxt_output)
                    out_file.write("\n")
