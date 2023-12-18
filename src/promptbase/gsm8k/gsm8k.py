# generate.py
import json
import pathlib

from promptbase.utils.helpers import text_completion, run_batch_jobs
from datasets import load_dataset


my_path = pathlib.Path(__file__).parent.resolve()


def extract_substrings(text):
    parts = text.split(r"\boxed")
    matches = []

    for part in parts[1:]:  # Skip the first part as it does not start with \boxed
        if part.startswith("{"):
            brace_level = 0
            for i, char in enumerate(part):
                if char == "{":
                    brace_level += 1
                elif char == "}":
                    brace_level -= 1
                    if brace_level == 0:
                        matches.append(
                            part[1:i]
                        )  # Extract the content inside the braces
                        break

    if len(matches) == 0:
        return None

    return matches[0]


def solve(task):
    idx, prompt = task

    for retry in range(5):
        response = text_completion(
            prompt=prompt,
            max_tokens=1200 + retry * 500,
            log_file="gsm8k.log",
            max_trial=5,
            temperature=retry * 0.5,
            model="gpt-4-1106-preview",
        )

        if not response["success"]:
            answer = None
            text = None
        else:
            text = response["text"]
            answer = extract_substrings(text)

        if answer:
            break

    if answer:
        with open(my_path.parent / "generations" / "gsm8k.jsonl", "a") as f:
            f.write(json.dumps({"idx": idx, "answer": answer, "proof": text}) + "\n")


def generate():
    ds = load_dataset("gsm8k", "main")["test"]
    tasks = []
    for idx, row in enumerate(ds):
        prompt = (
            row["question"]
            + "\nPlease end your solution with Answer: $\\boxed{number}$ where number is the numerical answer without unit.\nSolution:"
        )
        tasks.append((idx, prompt))
    run_batch_jobs(solve, tasks, max_thread=20)


def evaluate():
    rows = []
    ds = load_dataset("gsm8k", "main")["test"]
    with open(my_path.parent / "generations" / "gsm8k.jsonl", "r") as f:
        for line in f:
            row = json.loads(line)
            row["answer"] = extract_substrings(row["proof"])
            rows.append(row)

    def check_answer(official, student):
        return abs(official - student) < (abs(official) + 1e-6) * 1e-6

    n_correct = 0
    for i, row in enumerate(rows):
        idx = row["idx"]
        gpt_answer = None
        official_answer = None
        official_answer = ds[idx]["answer"].split("####")[1].replace(",", "")

        try:
            gpt_answer = (
                row["answer"].replace(",", "").split("\n## ")[0].replace("\%", "")
            )

            if gpt_answer == official_answer:
                n_correct += 1
                continue

            official_float = float(official_answer)
            gpt_float = float(gpt_answer)
            n_correct += check_answer(official_float, gpt_float)
            continue
        except:
            with open("parse.txt", "a") as f:
                f.write("=" * 80 + "\n")
                f.write(f"idx:{idx}\n")
                f.write("official_answer:" + str(official_answer) + "\n")
                f.write("gpt_answer:" + str(gpt_answer) + "\n")
                f.write("-" * 40 + "\n")
                f.write(ds[idx]["answer"] + "\n")
                f.write("-" * 40 + "\n")
                f.write(row["proof"] + "\n")

    print(
        "n_correct:",
        n_correct,
        "n_total:",
        len(rows),
        "accuracy:",
        n_correct / len(rows),
    )
