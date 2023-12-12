# Generate
import sys, json, re, traceback, hashlib, math
from promptbase import utils
from datasets import Dataset
from collections import Counter

prompts = []
chat_mode = False
ds = None

def fetch_data():
    global prompts
    global ds
    data_file = utils.fetch_dataset_blob("humaneval")
    ds = Dataset.from_file(data_file)
    for row in ds:
        if chat_mode:
            prompt = (
                row["prompt"]
                + "\n\nPlease complete the function above together with the function header."
            )
        else:
            prompt = (
                "## Here is the official solution of one python exercise via only one function:\n"
                + row["prompt"]
            )  # 118
            # prompt = f"## Solution of the coding exercise `{row['entry_point']}`:\n" + row["prompt"]
            # prompt = f"## Official solution of the coding exercise `{row['entry_point']}`:\n" + row["prompt"]
        prompts.append(prompt)


def extract_substrings(text):
    return re.findall(r"```(.*?)```", text, re.DOTALL)


def solve(idx):
    global prompts

    for retry in range(5):
        response = utils.text_completion(
            prompt=prompts[idx],
            max_tokens=600,
            log_file="human_eval.log",
            max_trial=5,
            temperature=retry * 0.05,
            model="gpt-4-1106-preview",
            stop=["##"],
        )

        if not response["success"]:
            code = None
        else:
            if chat_mode:
                text = response["text"]
                substrings = extract_substrings(text)
                substrings = [s for s in substrings if "def " in s]
                code = max(substrings, key=len, default="") if substrings else None
            else:
                code = prompts[idx] + response["text"]

        if code:
            break

    if code:
        with open("gpt4.jsonl", "a") as f:
            f.write(json.dumps({"idx": idx, "code": code}) + "\n")


def generate():
    utils.run_batch_jobs(solve, range(len(prompts)), max_thread=20)


def evaluate():
    # open gpt4.jsonl
    rows = []
    with open("gpt4_text_fixed.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))

    env = {
        "hashlib": hashlib,
        "re": re,
        "Counter": Counter,
        "factorial": math.factorial,
    }
    n_success = 0
    for row in rows:
        code = row["code"]
        if code.startswith("python"):
            code = code[6:]
        code = (
            code.split("# Test")[0]
            .split("# test")[0]
            .split("\nprint")[0]
            .split("\nassert")[0]
            .split("# END")[0]
            .split("<|ipynb_marker|>")[0]
            .split("\n# Check your answer")[0]
        )
        code += (
            "\n"
            + ds[row["idx"]]["test"]
            + "\ncheck("
            + ds[row["idx"]]["entry_point"]
            + ")"
        )

        try:
            exec(code, env, env)
            n_success += 1
        except Exception as e:
            err = traceback.format_exc()
            if "AssertionError" not in err:
                print(traceback.format_exc())
                print(code)
                print("=" * 100)
            n_success += 0

    print("Number of successes:", n_success)
    print("Number of rows:", len(rows))
    print("Success rate:", n_success / len(rows))
