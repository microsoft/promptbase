import dataclasses
import json
import pathlib

from typing import Any

import joblib


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
