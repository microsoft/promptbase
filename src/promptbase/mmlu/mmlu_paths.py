import pathlib

_my_path = pathlib.Path(__file__).parent.resolve()

mmlu_data_dir = _my_path.parent / "datasets" / "mmlu"

mmlu_generations_dir = _my_path.parent / "generations"
