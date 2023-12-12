import os

import promptbase.mmlu.MMLU as MMLU
from promptbase.mmlu.embed_problems import *
from promptbase.mmlu.problem_utils import *

dev_problem = "MMLU_dev"
test_problem = "MMLU_test"

model_name = "gpt-4-1106-preview"

if not os.path.exists(problem_files[dev_problem] + ".json.gz"):
    embed_file(problem_files[dev_problem] + ".json")

if not os.path.exists(problem_files[test_problem] + ".json.gz"):
    embed_file(problem_files[test_problem] + ".json")

MMLU.generate_solutions_without_rank(dev_problem, run_name=f"{dev_problem}/cot", model=model_name)
MMLU.run_cot_without_rank(test_problem, run_name=f"{test_problem}/cot_knn",
            examples=f"expt/{dev_problem}/cot/result", mode = "knn", 
            num_examples=5, num_repeat=5, max_thread=50, model=model_name)
MMLU.run_cot_without_rank(test_problem, run_name=f"{test_problem}/cot_via_knn",
            examples=f"expt/{test_problem}/cot_knn/result", mode = "knn", 
            num_examples=5, num_repeat=15, max_thread=50, model=model_name)
MMLU.run_logprobs(test_problem, run_name=f"{test_problem}/logprobs5",
            num_examples=5, num_repeat=10,  max_thread=50, model=model_name)
