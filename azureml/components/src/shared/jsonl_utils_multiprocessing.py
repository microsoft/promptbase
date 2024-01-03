import dataclasses
import json
import multiprocessing
import pathlib
import queue
import tempfile
import time

from typing import Any, Callable

from .jsonl_file_utils import JSONLReader, JSONLWriter
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
    with JSONLReader(source_file, source_encoding) as in_file:
        for nxt in in_file:
            logger.info(f"Reading line {lines_read}")
            target_queue.put(nxt)
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

    with JSONLWriter(dest_file, dest_encoding) as out_file:
        while n_complete_markers_seen < n_complete_markers_expected:
            nxt_item = target_queue.get()
            if isinstance(nxt_item, _WorkCompleteMarker):
                logger.info(f"Got WorkCompleteMarker '{nxt_item.message}'")
                n_complete_markers_seen += 1
            else:
                logger.info(f"Writing item")
                out_file.write_line(nxt_item)


def _error_to_jsonl_worker(
    *,
    error_file: pathlib.Path | None,
    error_encoding: str | None,
    target_queue: multiprocessing.Queue,
    n_complete_markers_expected: int,
    n_errors_max: int,
):
    logger = get_logger_for_process(__file__, f"error")
    n_complete_markers_seen = 0
    n_errors_seen = 0

    with JSONLWriter(error_file, error_encoding) as err_file:
        while n_complete_markers_seen < n_complete_markers_expected:
            nxt_item = target_queue.get()

            if isinstance(nxt_item, _WorkCompleteMarker):
                logger.info(f"Got WorkCompleteMarker '{nxt_item.message}'")
                n_complete_markers_seen += 1
            else:
                n_errors_seen += 1
                logger.warning(f"Received Error Item (total={n_errors_seen})")
                err_file.write_line(nxt_item)

            if n_errors_seen > n_errors_max:
                logger.fatal(f"Error limit of {n_errors_max} exceeded")
                logger.fatal(f"Final item: {nxt_item}")
                # This will kill the process
                raise ValueError(
                    f"Too many error items ({n_errors_seen} > {n_errors_max})"
                )
        logger.info("About to close error file")


def _monitor_worker(
    *,
    source_queue: multiprocessing.Queue,
    dest_queue: multiprocessing.Queue,
    worker_time_queue: multiprocessing.Queue,
    n_complete_markers_expected: int,
):
    logger = get_logger_for_process(__file__, f"monitor")
    UPDATE_SECS = 30
    logger.info("Starting")
    all_times = []

    n_complete_markers_seen = 0
    while n_complete_markers_seen < n_complete_markers_expected:
        time.sleep(UPDATE_SECS / (1 + n_complete_markers_seen))
        src_count = source_queue.qsize()
        dst_count = dest_queue.qsize()

        # Since qsize() is not reliable for multiprocessing, have a
        # slightly unpleasant pattern here
        try:
            while True:
                fetched = worker_time_queue.get_nowait()
                if isinstance(fetched, _WorkCompleteMarker):
                    n_complete_markers_seen += 1
                else:
                    all_times.append(fetched)
        except queue.Empty:
            pass

        min_time = -1
        max_time = -1
        mean_time = -1
        if len(all_times) > 0:
            min_time = min(all_times)
            max_time = max(all_times)
            mean_time = sum(all_times) / len(all_times)

        logger.info(f"Items in Source Queue: {src_count}")
        logger.info(f"Items in Destination Queue: {dst_count}")
        logger.info(f"Items processed so far: {len(all_times)}")
        logger.info(
            f"Times: {min_time:.2f}s (min) {mean_time:.2f}s (mean) {max_time:.2f}s (max)"
        )
    logger.info("Completed")


def _queue_worker(
    *,
    map_func: Callable[[dict[str, Any]], dict[str, Any] | None],
    source_queue: multiprocessing.Queue,
    dest_queue: multiprocessing.Queue,
    error_queue: multiprocessing.Queue,
    run_stats_queue: multiprocessing.Queue,
    worker_time_queue: multiprocessing.Queue,
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
            start_time = time.time()
            try:
                nxt_result = map_func(nxt_item)
                stop_time = time.time()
                if nxt_result is not None:
                    dest_queue.put(nxt_result)
                else:
                    logger.info("map_func returned None")
                success_count += 1
            except Exception as e:
                stop_time = time.time()
                logger.warn(f"Item failed: {e}")
                error_queue.put(nxt_item)
                failure_count += 1
            worker_time_queue.put(stop_time - start_time)
    logger.info(f"Completed work items")
    marker = _WorkCompleteMarker(f"queue_worker{id:02}")
    dest_queue.put(marker)
    error_queue.put(marker)
    worker_time_queue.put(marker)
    stats = RunStats(success_count=success_count, failure_count=failure_count)
    run_stats_queue.put(stats)
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

    run_stats_queue = multiprocessing.Queue(maxsize=n_worker_tasks)
    timing_queue = multiprocessing.Queue()

    # List of the various processes spawned
    # This will _not_ include the error output worker

    worker_processes = []

    # Setup the enqueuer
    enqueue_process = multiprocessing.Process(
        target=_enqueue_from_jsonl_worker,
        kwargs=dict(
            source_file=source_file,
            source_encoding=source_encoding,
            target_queue=source_queue,
            n_complete_markers=n_worker_tasks,
        ),
        name="Enqueuer",
    )
    worker_processes.append(enqueue_process)

    # Setup the workers
    for i in range(n_worker_tasks):
        nxt = multiprocessing.Process(
            target=_queue_worker,
            kwargs=dict(
                map_func=map_func,
                source_queue=source_queue,
                dest_queue=dest_queue,
                error_queue=error_queue,
                run_stats_queue=run_stats_queue,
                worker_time_queue=timing_queue,
                id=i,
            ),
            name=f"Worker {i}",
        )
        worker_processes.append(nxt)

    # Setup  the monitor
    monitor_process = multiprocessing.Process(
        target=_monitor_worker,
        kwargs=dict(
            source_queue=source_queue,
            dest_queue=dest_queue,
            worker_time_queue=timing_queue,
            n_complete_markers_expected=n_worker_tasks,
        ),
        name="Monitor",
    )
    worker_processes.append(monitor_process)

    # Setup the output dequeuer
    dequeue_output_process = multiprocessing.Process(
        target=_dequeue_to_jsonl_worker,
        kwargs=dict(
            dest_file=dest_file,
            dest_encoding=dest_encoding,
            target_queue=dest_queue,
            n_complete_markers_expected=n_worker_tasks,
        ),
        name="Output",
    )
    worker_processes.append(dequeue_output_process)

    # Start the error dequeuer
    dequeue_error_output_process = multiprocessing.Process(
        target=_error_to_jsonl_worker,
        kwargs=dict(
            error_file=error_file,
            error_encoding=error_encoding,
            target_queue=error_queue,
            n_complete_markers_expected=n_worker_tasks,
            n_errors_max=n_errors_max,
        ),
        name="Error Output",
    )
    dequeue_error_output_process.start()

    # Start the workers
    for wp in worker_processes:
        wp.start()

    # Wait for processes to complete

    # Check on errors first, since we may want to kill everything
    dequeue_error_output_process.join()
    if dequeue_error_output_process.exitcode != 0:
        _logger.critical(
            f"Detected non-zero exit from dequeue_error_output_process: {dequeue_error_output_process.exitcode}"
        )
        for wp in worker_processes:
            wp.kill()
        _logger.critical("Worker processes terminated")
        raise Exception("Too many errors. See log for details")

    # Do a normal exit
    _logger.info("Joining workers")
    for wp in worker_processes:
        wp.join()

    total_successes = 0
    total_failures = 0
    for _ in range(n_worker_tasks):
        nxt: RunStats = run_stats_queue.get()
        total_successes += nxt.success_count
        total_failures += nxt.failure_count

    _logger.info(f"Total Successful items: {total_successes}")
    _logger.info(f"Total Failed items    : {total_failures}")
    _logger.info("line_map_mp completed")
