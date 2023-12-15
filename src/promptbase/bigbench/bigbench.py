from .bigbench_cot import process_cot
from .bigbench_score import score
from .bigbench_answer import process_answers
from promptbase.utils.consts import BIGBENCH_SUBJECTS

def generate(subject  = "all"):
  if subject != "all" and subject not in BIGBENCH_SUBJECTS:
    print(f"Invalid subject: {subject}")
    return
  print(f"Running BigBench generation for subject {subject}")
  process_cot(subject)
  process_answers(subject)

def evaluate():
  score()