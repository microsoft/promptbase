from .bigbench_cot import process_cot
from .bigbench_score import score
from .bigbench_answer import process_answers
from promptbase.bigbench.consts import BIGBENCH_SUBJECTS

def generate(subject: str, overwrite: bool):
  if subject != "all" and subject not in BIGBENCH_SUBJECTS:
    print(f"Invalid subject: {subject}")
    return
  print(f"Running BigBench generation for subject {subject}")
  process_cot(subject, overwrite)
  process_answers(subject, overwrite)

def evaluate():
  score()