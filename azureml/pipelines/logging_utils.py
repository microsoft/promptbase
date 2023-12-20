import logging
import pathlib


def get_standard_logger_for_file(file_path: str) -> logging.Logger:
    _logger = logging.getLogger(pathlib.Path(file_path).name)
    _logger.setLevel(logging.INFO)
    return _logger
