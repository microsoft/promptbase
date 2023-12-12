from .bigbench_cot import process_cot
from .bigbench_score import score
from .bigbench_answer import process_answers

def generate():
  process_cot(test_name="all")
  process_answers(test_name="all")

def evaluate():
  score()