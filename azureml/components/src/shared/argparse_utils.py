import json

from .logging_utils import get_standard_logger_for_file


_logger = get_standard_logger_for_file(__file__)


def json_loads_fixer(cmd_line_arg: str) -> dict[str, any]:
    """Parses JSON command line arguments which have acquired extra quotes."""
    _logger.info(f"Attempting to parse: {cmd_line_arg}")
    if cmd_line_arg.startswith("'") and cmd_line_arg.endswith("'"):
        _logger.info("Detected quotes")
        cmd_line_arg = cmd_line_arg[1:-1]
        _logger.info(f"Trimmed argument: {cmd_line_arg}")
    return json.loads(cmd_line_arg)
