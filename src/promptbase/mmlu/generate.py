import os
import pathlib

from . import MMLU
from .embed_problems import embed_file
from .mmlu_paths import mmlu_data_dir, mmlu_generations_dir

model_name = "gpt-4-1106-preview"


def generate(dataset_name: str):
    dev_problem = f"mmlu_{dataset_name}_val"
    test_problem = f"mmlu_{dataset_name}_test"

    if not os.path.exists(str(mmlu_data_dir / dev_problem) + ".json.gz"):
        embed_file(str(mmlu_data_dir / dev_problem) + ".json")

    if not os.path.exists(str(mmlu_data_dir / test_problem) + ".json.gz"):
        embed_file(str(mmlu_data_dir / test_problem) + ".json")

    MMLU.generate_solutions_without_rank(
        dev_problem, run_name=f"{dev_problem}/cot", model=model_name
    )
    MMLU.run_cot_without_rank(
        test_problem,
        run_name=f"{test_problem}/cot_knn",
        examples=str(
            mmlu_generations_dir / f"expt" / f"{dev_problem}" / "cot" / "result"
        ),
        mode="knn",
        num_examples=5,
        num_repeat=5,
        max_thread=50,
        model=model_name,
    )
    MMLU.run_cot_without_rank(
        test_problem,
        run_name=f"{test_problem}/cot_via_knn",
        examples=str(
            mmlu_generations_dir / f"expt" / f"{test_problem}" / "cot_knn" / "result"
        ),
        mode="knn",
        num_examples=5,
        num_repeat=15,
        max_thread=50,
        model=model_name,
    )
    if False:
        # Logprobs not currently available in OpenAI API
        MMLU.run_logprobs(
            test_problem,
            run_name=f"{test_problem}/logprobs5",
            num_examples=5,
            num_repeat=10,
            max_thread=50,
            model=model_name,
        )
