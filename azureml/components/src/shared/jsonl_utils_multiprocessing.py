import json
import multiprocessing
import pathlib

from typing import Any, Callable

from .logging_utils import get_standard_logger_for_file, get_logger_for_process


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


def _queue_worker(
        *,
        map_func: Callable[[dict[str, Any]], dict[str, Any] | None],
        source_queue: multiprocessing.Queue,
        dest_queue: multiprocessing.Queue,
        id: int,
):
    logger = get_logger_for_process(__file__, f"worker{id:02}")

    done = False
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
                    pass
                else:
                    logger.info("map_func returned None")
            except Exception as e:
                logger.warn(f"Item failed")
    logger.info(f"Completed work items")


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
):
    _logger.info("Starting line_map_mp")

    assert source_file.exists()

    source_queue = multiprocessing.Queue(maxsize=2 * n_worker_tasks)
    dest_queue = multiprocessing.Queue(maxsize=2 * n_worker_tasks)

    enqueue_process = multiprocessing.Process(
        target=_enqueue_from_jsonl_worker,
        kwargs=dict(
            source_file=source_file,
            source_encoding=source_encoding,
            target_queue=source_queue,
            n_complete_markers=n_worker_tasks,
        ),
    )

    worker_processes = []
    for i in range(n_worker_tasks):
        nxt = multiprocessing.Process(
            target=_queue_worker,
            kwargs = dict(map_func=map_func,source_queue=source_queue, dest_queue=dest_queue, id=i)
        )
        nxt.start()
        worker_processes.append(nxt)


    enqueue_process.start()

    enqueue_process.join()  
    for wp in worker_processes:
        wp.join()
