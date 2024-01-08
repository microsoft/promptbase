from .problem_utils import *
import gzip


def load_problems(file_name):
    with gzip.open(file_name + ".json.gz", "rt") as f:
        problems = json.loads(f.read())
    return problems


# Load problems from the file
problems = load_problems(f"expt/final/MMLU_medical_genetics/logits0/result")

# Compute statistics on the loaded problems
summary = compute_statistics(problems)
print(summary)
