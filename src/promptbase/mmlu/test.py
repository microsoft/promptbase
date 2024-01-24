import multiprocessing
import os

from . import MMLU
from .embed_problems import *
from .problem_utils import *

dev_name = "MMLU_dev"
test_name = "MMLU_test"
dev_name = "MMLU_chemistry"
test_name = "MMLU_chemistry"

# embed questions
if not os.path.exists(problem_files[dev_name] + ".json.gz"):
    embed_file(problem_files[dev_name] + ".json")

if not os.path.exists(problem_files[test_name] + ".json.gz"):
    embed_file(problem_files[test_name] + ".json")

# generate cot solutions on dev set
if not os.path.exists(f"mmlu/expt/{dev_name}/cot/result.json.gz"):
    MMLU.run_cot(dev_name, example_selector="random", max_thread=50)

# generate cot solutions on test set via dev set
if not os.path.exists(f"mmlu/expt/{test_name}/cot_merged.json.gz"):

    def generate_test_cot_initial(index):
        MMLU.run_cot(
            test_name,
            run_name=f"{test_name}/cot_{index}",
            examples=f"expt/{dev_name}/cot/result",
            num_repeat=1,
            max_thread=30,
            num_examples=5,
            example_selector="knn",
            model="gpt-4-1106-preview",
        )
        return "Done!"

    with multiprocessing.Pool(processes=5) as pool:
        results = pool.map(generate_test_cot_initial, range(5))

    cot_rows1 = load_problems(f"expt/{test_name}/cot_0/result")
    cot_rows2 = load_problems(f"expt/{test_name}/cot_1/result")
    cot_rows3 = load_problems(f"expt/{test_name}/cot_2/result")
    cot_rows4 = load_problems(f"expt/{test_name}/cot_3/result")
    cot_rows5 = load_problems(f"expt/{test_name}/cot_4/result")

    def merge_ds(dataset_list):
        cot_rows = {}
        for rows_set in dataset_list:
            for row in rows_set:
                if row["question_number"] not in cot_rows:
                    cot_rows[row["question_number"]] = copy.copy(row)
                    cot_rows[row["question_number"]]["expt"] = {}
                for key in row["expt"]:
                    cot_rows[row["question_number"]]["expt"][key] = row["expt"][key]
        return list(cot_rows.values())

    cot_rows = merge_ds([cot_rows1, cot_rows2, cot_rows3, cot_rows4, cot_rows5])
    save_problems(f"expt/{test_name}/cot_merged", cot_rows)


# solutions on test set


## generate cot solutions on test set via test set
def generate_test_cot(index):
    MMLU.run_cot_without_rank(
        test_name,
        run_name=f"{test_name}/cot_via_test_{index}_v8",
        examples=f"mmlu/expt/{test_name}/cot_merged",
        num_repeat=1,
        max_thread=30,
        num_examples=5,
        mode="knn",
        model="gpt-4-1106-preview",
    )
    return "Done!"


with multiprocessing.Pool(processes=15) as pool:
    results = pool.map(generate_test_cot, range(5))
