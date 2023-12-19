import json
import os
import pathlib
import re
import string
import zipfile

from textwrap import dedent

import requests

from promptbase.utils.helpers import (
    fetch_dataset_blob,
    get_standard_logger_for_file,
    run_batch_jobs,
    text_completion,
)

_logger = get_standard_logger_for_file(__file__)

CHAT_MODE = True
DROP_DATASET_PATH = "datasets/DROP/drop_dataset_dev.json"


def extract_valid_answers(validated_answers):
    answers = []
    for answer in validated_answers:
        if answer["number"] != "":
            answer_type = "number"
            answers.append(["number", answer["number"]])

        elif set(answer["date"].values()) != {""}:
            answer_type = "date"
            answers.append(["date", answer["date"]])

        elif answer["spans"] != []:
            answer_type = "span"
            answers.append(["span", answer["spans"]])

        else:
            raise Exception("Invalid answer type")

    return answers


computed_idxs = set()
prompts = []
answers = []


def fetch_data():
    global prompts
    global answers
    global computed_idxs
    _logger.info("Starting fetch_data")
    zip_file = pathlib.Path(fetch_dataset_blob("drop"))

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(zip_file.parent)
    _logger.info("Data unzipped")

    dataset_path = zip_file.parent / "drop_dataset" / "drop_dataset_dev.json"

    with open(dataset_path, "r") as f:
        data = json.load(f)
    _logger.info("Loaded JSON data")

    for key in data:
        passage = data[key]["passage"]
        for qa in data[key]["qa_pairs"]:
            prompt_question = qa["question"]
            answers.append(extract_valid_answers(qa["validated_answers"]))
            if CHAT_MODE:
                prompt = [
                    {
                        "role": "system",
                        "content": dedent(
                            """\
                Answer the following reading comprehension **Question** based on the **Passage** below.
                First, think step by step and write an **Explanation** for reasoning through the question.
                Then, when prompted by the user for a **Final Answer**, analyze your explanation and write just the **Final Answer** succinctly using as few words as possible. A good final answer may just be a Name, Place, Date, Number, etc. and does not have additional explanatory text. You should specify numbers in numeric form (e.g 1, 2, 3) and not alphabetical (one, two, three). Do not say the final answer until the user asks for it."""
                        ),
                    },
                    {
                        "role": "user",
                        "content": dedent(
                            f"""\
                **Passage:** {passage}
                ----
                **Question:** {prompt_question}
                ----
                **Explanation**:"""
                        ),
                    },
                ]
                prompts.append(prompt)
            else:
                prompt = dedent(
                    f"""\
                Answer the following reading comprehension **Question** based on the **Passage** below.
                First, think step by step and write an **Explanation** for reasoning through the question.
                Then, analyze your explanation and write just the **Final Answer** succinctly using as few words as possible. A good final answer may just be a Name, Place, Date, Number, etc. and does not have additional explanatory text. You should specify numbers in numeric form (e.g 1, 2, 3) and not alphabetical (one, two, three).
                ----
                **Passage:** {passage}
                ----
                **Question:** {prompt_question}
                ----
                **Explanation**: """
                )
                prompts.append(prompt)

    if CHAT_MODE:
        computed_idxs = set()
        if os.path.isfile("drops_cot_raw_responses_chat.jsonl"):
            with open("drops_cot_raw_responses_chat.jsonl", "r") as f:
                computed_idxs = set([json.loads(line)["idx"] for line in f])
    else:
        computed_idxs = set()
        if os.path.isfile("drops_cot_raw_responses.jsonl"):
            with open("drops_cot_raw_responses.jsonl", "r") as f:
                computed_idxs = set([json.loads(line)["idx"] for line in f])
    _logger.info("fetch_data complete")


def extract_substrings(text):
    return re.findall(r"```(.*?)```", text, re.DOTALL)


def solve(idx):
    global prompts
    global answers
    global computed_idxs
    global CHAT_MODE
    _logger.info("Starting solve")

    if CHAT_MODE:
        model_name = "gpt-4-1106-preview"
        reasoning_file_name = "drop_data_cot_reasoning_chat.log"
        answer_file_name = "drop_data_cot_final_chat.log"
        json_file_name = "drops_cot_raw_responses_chat.jsonl"
    else:
        model_name = "gpt-4-1106-preview"
        reasoning_file_name = "drop_data_cot_reasoning.log"
        answer_file_name = "drop_data_cot_final.log"
        json_file_name = "drops_cot_raw_responses.jsonl"

    if idx in computed_idxs:
        return

    for retry in range(5):
        response = text_completion(
            prompt=prompts[idx],
            model=model_name,
            max_tokens=500,
            log_file=reasoning_file_name,
            max_trial=5,
            temperature=0,
            stop=["**", "--", "\n"],
        )

        if not response["success"]:
            code = None
        else:
            text = response["text"]
            substrings = extract_substrings(text)
            substrings = [s for s in substrings if "def " in s]
            code = max(substrings, key=len, default="") if substrings else None

        if code:
            break

        # Add the response content into the prompt
        if CHAT_MODE:
            prompts[idx].append(
                {
                    "role": "assistant",
                    "content": response["response"]["choices"][0]["text"],
                }
            )
            prompts[idx].append({"role": "user", "content": "----\n**Final Answer**: "})
        else:
            prompts[idx] += response["response"]["choices"][0]["text"]
            prompts[idx] += "----\n**Final Answer**: "

        response_final = text_completion(
            prompt=prompts[idx],
            model=model_name,
            max_tokens=100,
            log_file=answer_file_name,
            max_trial=5,
            temperature=0,
            stop=["**", "--", "\n"],
        )

        if not response_final["success"]:
            code = None
        else:
            text = response_final["text"]
            substrings = extract_substrings(text)
            substrings = [s for s in substrings if "def " in s]
            code = max(substrings, key=len, default="") if substrings else None

        if code:
            break

    with open(json_file_name, "a") as f:
        f.write(
            json.dumps(
                {
                    "idx": idx,
                    "response": response_final["response"]["choices"][0]["text"],
                    "answers": answers[idx],
                }
            )
            + "\n"
        )


def read_jsonl(file):
    with open(file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return [json.loads(l) for l in lines]


def calculate_accuracy(responses):
    total_count = len(responses)
    match_count = 0

    for response in responses:
        for answer in response["answers"]:
            answer_type = answer[0]
            answer_value = answer[1]
            if answer_type == "span":
                if check_span(response["response"], answer_value):
                    match_count += 1
                    break
            elif answer_type == "number":
                if check_number(response["response"], answer_value):
                    match_count += 1
                    break
            elif answer_type == "date":
                if check_date(response["response"], answer_value):
                    match_count += 1
                    break
    return float(match_count) / float(total_count)


def check_span(response, answer_spans):
    # Removing punctuations and converting to lower case
    response = "".join(c for c in response.lower() if c not in string.punctuation)

    response_words = set(response.split())

    for answer_span in answer_spans:
        answer_span = "".join(
            c for c in answer_span.lower() if c not in string.punctuation
        )
        answer_words = set(answer_span.split())

        if answer_words.issubset(response_words):
            return True

    print(response, "|", answer_spans)
    return False


def check_number(response, answer_number):
    # Find all numeric substrings (including decimals and commas) in the response.
    response_numbers = re.findall(r"\d[\d,]*(?:\.\d+)?|\.\d+", response)

    # Convert the response and the answer into floats before comparison
    return any(
        float(num.replace(",", "")) == float(answer_number.replace(",", ""))
        for num in response_numbers
    )


def check_date(response, answer_date):
    response_parts = response.split()

    # Special case: If both day and month are specified in the answer, both must be correct
    if answer_date["day"] and answer_date["month"]:
        try:
            day_position = response_parts.index(answer_date["day"])
            month_position = response_parts.index(answer_date["month"].lower())
            # If the date is provided as "month day", or "day month".
            if abs(day_position - month_position) == 1:
                response_parts.remove(answer_date["day"])
                response_parts.remove(answer_date["month"].lower())
            else:
                return False
        except ValueError:
            return False

    date_parts = ["day", "month", "year"]

    for part in response_parts:
        for date_part in date_parts:
            # If date_part is specified in the answer.
            if answer_date[date_part]:
                # If it's a year, match it with a 4-digit number.
                if date_part == "year":
                    if (
                        re.match(r"\b(\d{4})\b", part)
                        and part == answer_date[date_part]
                    ):
                        # print(response_parts, "|", answer_date)
                        return True
                # Otherwise, match the words irrespective of case.
                else:
                    if part.lower() == answer_date[date_part].lower():
                        # print(response_parts, "|", answer_date)
                        return True

    # print(response_parts, "|", answer_date)
    return False


def generate():
    _logger.info("Starting generate")
    fetch_data()
    _logger.info("Beginning batch jobs")
    run_batch_jobs(solve, range(len(prompts)), max_thread=20)


def evaluate():
    responses = read_jsonl("drops_cot_raw_responses_chat.jsonl")  # your jsonl filename
    accuracy = calculate_accuracy(responses)
    print("Accuracy: ", accuracy)
