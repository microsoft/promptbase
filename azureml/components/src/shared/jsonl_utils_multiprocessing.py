import dataclasses
import json
import multiprocessing
import pathlib
import tempfile

from typing import Any, Callable

from .logging_utils import get_standard_logger_for_file, get_logger_for_process


_logger = get_standard_logger_for_file(__file__)


class _WorkCompleteMarker:
    def __init__(self, message: str):
        self._message = message

    @property
    def message(self) -> str:
        return self._message


@dataclasses.dataclass
class RunStats:
    success_count: int = int()
    failure_count: int = int()


def _enqueue_from_jsonl_worker(
    *,
    source_file: pathlib.Path,
    source_encoding: str,
    target_queue: multiprocessing.Queue,
    n_complete_markers: int,
):
    logger = get_logger_for_process(__file__, "enqueue")

    lines_read = 0
    with open(source_file, "r", encoding=source_encoding) as in_file:
        for nxt in in_file:
            logger.info(f"Reading line {lines_read}")
            nxt_dict = json.loads(nxt)
            target_queue.put(nxt_dict)
            lines_read += 1

    for i in range(n_complete_markers):
        logger.info(f"WorkerCompleteMarker {i}")
        nxt_marker = _WorkCompleteMarker(f"Completion marker {i}")
        target_queue.put(nxt_marker)
    logger.info(f"Completed")


def _dequeue_to_jsonl_worker(
    *,
    dest_file: pathlib.Path,
    dest_encoding: str,
    target_queue: multiprocessing.Queue,
    n_complete_markers_expected: int,
):
    logger = get_logger_for_process(__file__, f"output")

    n_complete_markers_seen = 0

    with open(dest_file, "w", encoding=dest_encoding) as out_file:
        while n_complete_markers_seen < n_complete_markers_expected:
            nxt_item = target_queue.get()
            if isinstance(nxt_item, _WorkCompleteMarker):
                logger.info(f"Got WorkCompleteMarker '{nxt_item.message}'")
                n_complete_markers_seen += 1
            else:
                logger.info(f"Writing item")
                nxt_output = json.dumps(nxt_item)
                out_file.write(nxt_output)
                out_file.write("\n")


def _error_to_jsonl_worker(
    *,
    error_file: pathlib.Path | None,
    error_encoding: str | None,
    target_queue: multiprocessing.Queue,
    n_complete_markers_expected: int,
    n_errors_max: int,
):
    logger = get_logger_for_process(__file__, f"error")

    def get_error_file(error_file_path: pathlib.Path | None):
        if error_file_path:
            return open(error_file_path, "a", encoding=error_encoding)
        else:
            return tempfile.TemporaryFile(mode="w", encoding="utf-8-sig")

    n_complete_markers_seen = 0
    n_errors_seen = 0

    with get_error_file(error_file) as err_file:
        while n_complete_markers_seen < n_complete_markers_expected:
            nxt_item = target_queue.get()

            if isinstance(nxt_item, _WorkCompleteMarker):
                logger.info(f"Got WorkCompleteMarker '{nxt_item.message}'")
                n_complete_markers_seen += 1
            else:
                n_errors_seen += 1
                logger.warning(f"Received Error Item (total={n_errors_seen})")
                nxt_output = json.dumps(nxt_item)
                err_file.write(nxt_output)
                err_file.write("\n")

            if n_errors_seen > n_errors_max:
                logger.fatal(f"Error limit of {n_errors_max} exceeded")
                raise ValueError("Too many error items")


def _queue_worker(
    *,
    map_func: Callable[[dict[str, Any]], dict[str, Any] | None],
    source_queue: multiprocessing.Queue,
    dest_queue: multiprocessing.Queue,
    error_queue: multiprocessing.Queue,
    id: int,
):
    logger = get_logger_for_process(__file__, f"worker{id:02}")

    done = False
    success_count = 0
    failure_count = 0
    while not done:
        nxt_item = source_queue.get()

        if isinstance(nxt_item, _WorkCompleteMarker):
            logger.info(f"Got WorkCompleteMarker '{nxt_item.message}'")
            done = True
        else:
            logger.info("Processing item")
            try:
                nxt_result = map_func(nxt_item)
                if nxt_result is not None:
                    dest_queue.put(nxt_result)
                else:
                    logger.info("map_func returned None")
                success_count += 1
            except Exception as e:
                logger.warn(f"Item failed")
                error_queue.put(nxt_item)
                failure_count += 1
    logger.info(f"Completed work items")
    marker = _WorkCompleteMarker(f"queue_worker{id:02}")
    dest_queue.put(marker)
    error_queue.put(marker)
    _logger.info(f"Exiting")


def line_map_mp(
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
    _logger.info("Starting line_map_mp")

    assert source_file.exists()

    source_queue = multiprocessing.Queue(maxsize=2 * n_worker_tasks)
    dest_queue = multiprocessing.Queue(maxsize=2 * n_worker_tasks)
    error_queue = multiprocessing.Queue(maxsize=2 * n_worker_tasks)

    # Start enqueuing items
    enqueue_process = multiprocessing.Process(
        target=_enqueue_from_jsonl_worker,
        kwargs=dict(
            source_file=source_file,
            source_encoding=source_encoding,
            target_queue=source_queue,
            n_complete_markers=n_worker_tasks,
        ),
    )
    enqueue_process.start()

    # Start the workers
    worker_processes = []
    for i in range(n_worker_tasks):
        nxt = multiprocessing.Process(
            target=_queue_worker,
            kwargs=dict(
                map_func=map_func,
                source_queue=source_queue,
                dest_queue=dest_queue,
                error_queue=error_queue,
                id=i,
            ),
        )
        nxt.start()
        worker_processes.append(nxt)

    # Start the dequeuers
    dequeue_output_process = multiprocessing.Process(
        target=_dequeue_to_jsonl_worker,
        kwargs=dict(
            dest_file=dest_file,
            dest_encoding=dest_encoding,
            target_queue=dest_queue,
            n_complete_markers_expected=n_worker_tasks,
        ),
    )
    dequeue_output_process.start()

    dequeue_error_output_process = multiprocessing.Process(
        target=_error_to_jsonl_worker,
        kwargs=dict(
            error_file=error_file,
            error_encoding=error_encoding,
            target_queue=error_queue,
            n_complete_markers_expected=n_worker_tasks,
            n_errors_max=n_errors_max,
        ),
    )
    dequeue_error_output_process.start()

    # Wait for processes to complete
    enqueue_process.join()
    for wp in worker_processes:
        wp.join()
    dequeue_output_process.join()
    dequeue_error_output_process.join()