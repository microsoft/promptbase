import logging
import pathlib
import sys

def get_standard_logger_for_file(file_path: str) -> logging.Logger:
    _logger = logging.getLogger(pathlib.Path(file_path).name)
    _logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s [%(levelname)s] : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    )
    _logger.addHandler(sh)
    return _logger