# generate
import sys, json, re
from promptbase.utils.helpers import run_batch_jobs, text_completion
from datasets import load_dataset

ds = None
prompts = []
rows = []

def fetch_data():
    global ds
    ds = load_dataset("hendrycks/competition_math")["test"]

    global prompts
    for row in ds:
        prompt = (
            row["problem"]
            + "\nPlease end your solution with Answer: $\\boxed{number}$ where number is the numerical answer without unit.\nSolution:"
        )
        prompts.append(prompt)


def fetch_data_2():
    global rows
    # open gpt4.jsonl
    with open("gpt4.jsonl") as f:
        for line in f:
            row = json.loads(line)
            row["answer"] = extract_substrings(row["proof"])
            rows.append(row)

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


def solve(idx):
    global prompts

    for retry in range(5):
        response = text_completion(
            prompt=prompts[idx],
            max_tokens=1200 + retry * 500,
            log_file="human_eval.log",
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
        with open("gpt4_2nd_round.jsonl", "a") as f:
            f.write(json.dumps({"idx": idx, "answer": answer, "proof": text}) + "\n")


def generate():
    run_batch_jobs(solve, range(len(prompts)), max_thread=20)


# parse


def check_answer(official, student):
    return abs(official - student) < (abs(official) + 1e-6) * 1e-6


def extract_and_convert_fraction(text):
    pattern = r"^\\frac\{(\d+)\}\{(\d+)\}$"
    match = re.match(pattern, text)
    if match:
        numerator, denominator = match.groups()
        return float(numerator) / float(denominator)

    pattern = r"^\\frac(\d)(\d)$"
    match = re.match(pattern, text)
    if match:
        numerator, denominator = match.groups()
        return float(numerator) / float(denominator)

    pattern = r"^-\\frac\{(\d+)\}\{(\d+)\}$"
    match = re.match(pattern, text)
    if match:
        numerator, denominator = match.groups()
        return -float(numerator) / float(denominator)

    pattern = r"^-\\frac(\d)(\d)$"
    match = re.match(pattern, text)
    if match:
        numerator, denominator = match.groups()
        return -float(numerator) / float(denominator)
    return text


def remove_latex_text_commands(text):
    """
    Removes all occurrences of \text{...} from a given LaTeX string.

    Parameters:
    text (str): The input LaTeX string.

    Returns:
    str: The LaTeX string with all \text{...} commands removed.
    """
    # Regular expression pattern to match \text{...}
    pattern = r"\\text\{.*?\}"

    # Replace all occurrences of \text{...} with an empty string
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)

    return cleaned_text


def evaluate():
    n_correct = 0
    for i, row in enumerate(rows):
        idx = row["idx"]
        gpt_answer = None
        official_answer = None

        official_answer = (
            extract_substrings(ds[idx]["solution"])
            .replace(" ", "")
            .replace("dfrac", "frac")
        )
        if official_answer.startswith(r"\$"):
            official_answer = official_answer[2:]

        try:
            gpt_answer = row["answer"].replace(" ", "").replace("dfrac", "frac")
            official_answer = remove_latex_text_commands(official_answer).replace(
                " ", ""
            )
            gpt_answer = remove_latex_text_commands(gpt_answer).replace(" ", "")
            official_answer = extract_and_convert_fraction(official_answer)
            gpt_answer = extract_and_convert_fraction(gpt_answer)

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
                f.write(ds[idx]["solution"] + "\n")
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
