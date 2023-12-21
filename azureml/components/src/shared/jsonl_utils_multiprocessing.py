import json
import multiprocessing
import pathlib

from typing import Any, Callable

from logging_utils import get_standard_logger_for_file, get_logger_for_process


_logger = get_standard_logger_for_file(__file__)


class _WorkCompleteMarker:
    def __init__(self, message: str):
        self._message = message

    @property
    def message(self) -> str:
        return self._message


def _enqueue_from_jsonl_worker(
    *,
    source_file: pathlib.Path,
    source_encoding: str,
    target_queue: multiprocessing.Queue,
    n_complete_markers: int,
):
    _logger = get_logger_for_process(__file__, "enqueue")


def line_map_mp(
    *,
    map_func: Callable[[dict[str, Any]], dict[str, Any]],
    source_file: pathlib.Path,
    source_encoding: str,
    dest_file: pathlib.Path,
    dest_encoding: str,
    n_worker_tasks: int,
    error_file: pathlib.Path | None = None,
    error_encoding: str | None = None,
):
    _logger.info("Starting line_map_mp")

    assert source_file.exists()

    source_queue = multiprocessing.Queue(max_size=2 * n_worker_tasks)
    dest_queue = multiprocessing.Queue(maxsize=2 * n_worker_tasks)
