import openai
import os
import json
import time
import threading
from promptbase.bigbench.consts import BIGBENCH_SUBJECTS
from promptbase.utils.helpers import text_completion, get_datasets_path, get_generations_path, get_standard_logger_for_file
from pathlib import Path


_logger = get_standard_logger_for_file(__file__)

def extract_chat_qa(few_shot_prompt):
    question = few_shot_prompt.split("\nA: ")[0].strip()
    answer = "A: " + few_shot_prompt.split("\nA: ")[1].strip()
    _logger.info("fewshot===")
    _logger.info("Q: %s", question)
    _logger.info("A: %s", answer)
    return (question, answer)


def do_chat_cot(bbh_test_path, cot_prompt_path, test_name, cot_results_path):
    _logger.info(f"Processing {test_name}")
    cot_results_filename = cot_results_path / f"{test_name}_chat_cot_results.json"
    if cot_results_filename.exists():
        test_results = json.load(open(cot_results_filename, "r"))
    else:
        test_results = []
    with open(cot_prompt_path, "r", encoding="utf-8") as file:
        file_contents = file.read()

        instruction_start_index = file_contents.find('-----')
        instruction_end_index = file_contents.find('Q:')
        offset_len = len('-----')

        if instruction_start_index != -1 and instruction_end_index != -1:
            instruction = file_contents[instruction_start_index+offset_len:instruction_end_index].strip()
        else:
            _logger.error(f"Error parsing bigbench cot-prompts in {cot_prompt_path}")
            instruction = "Answer the following question."

    few_shots = file_contents[instruction_end_index:].split("\n\n")
    # The first shot starts with an instruction, then two newlines, then the first shot
    qa_pairs = [extract_chat_qa(few_shot) for few_shot in few_shots[1:]]
    few_shot_messages = [
        {"role": "system", "content": f"{instruction}"},
    ]

    for question, answer in qa_pairs:
        few_shot_messages.append({"role": "user", "content": f"{question}"})
        few_shot_messages.append({"role": "assistant", "content": f"{answer}"})

    with open(bbh_test_path, "r", encoding="utf-8") as file:
        example_data = json.load(file)
        for i, example in enumerate(example_data["examples"]):
            _logger.info(
                f"Processing example {i} of {len(example_data['examples'])} for {test_name}"
            )
            if any([r["index"] == i for r in test_results]):
                _logger.info("Skipping example %s of test %s", i, test_name)
                continue
            prompt_messages = few_shot_messages + [
                {"role": "user", "content": "Q: " + example["input"]}
            ]
            response = text_completion(prompt=prompt_messages, max_tokens=2000, model="gpt-4-1106-preview", retry_wait=2, max_trial=int(1e9))
            if "text" not in response:
                _logger.error("Error in example %s of test %s response: %s", i, test_name, response)
                continue
            test_results.append(
                {
                    "index": i,
                    "test_name": test_name,
                    "prompt": prompt_messages,
                    "completion": response["text"]
                }
            )
            json.dump(test_results, open(cot_results_filename, "w"), indent=4)


def do_completion_cot(bbh_test_path, cot_prompt_path, test_name, cot_results_path):
    _logger.info("Processing %s", test_name)
    cot_results_filename = cot_results_path / f"{test_name}_completion_cot_results.json"
    if cot_results_filename.exists():
        test_results = json.load(open(cot_results_filename, "r"))
    else:
        test_results = []
    with open(cot_prompt_path, "r", encoding="utf-8") as file:
        cot_prompt_contents = file.read()
        # use everything starting with the third line
        cot_prompt_contents = "\n".join(cot_prompt_contents.split("\n")[2:]).strip()

    _logger.info("Chain of thought few-shot prompt: %s", cot_prompt_contents)

    with open(bbh_test_path, "r", encoding="utf-8") as file:
        example_data = json.load(file)
        for i, example in enumerate(example_data["examples"]):
            _logger.info(f"Processing example {i} of {len(example_data['examples'])} for {test_name}")
            if any([r["index"] == i for r in test_results]):
                _logger.info("Skipping example %s of test %s", i, test_name)
                continue
            prompt = f"{cot_prompt_contents}\n\nQ: {example['input']}\nA: Let's think step by step.\n"
            try:
                response = text_completion(prompt=prompt, max_tokens=2000, model="gpt-4-1106-comp", retry_wait=2, max_trial=int(1e9), stop="\n\n")
                test_results.append(
                    {
                        "index": i,
                        "test_name": test_name,
                        "prompt": prompt,
                        "completion": response["text"],
                    }
                )
            except Exception as e:
                _logger.warning("Caught exception: %s", e)
            cot_results_filename = cot_results_path / f"{test_name}_completion_cot_results.json"
            json.dump(
                test_results,
                open(cot_results_filename, "w"),
                indent=4,
            )


def process_cot(test_name: str, overwrite=False, api_type="chat"):
    _logger.info("Starting process_cot")
    if test_name == "all":
        subjects = BIGBENCH_SUBJECTS
    elif test_name in BIGBENCH_SUBJECTS:
        subjects = [test_name]
    else:
        _logger.error(f"Invalid test name: {test_name}")
        exit(1)

    bigbench_data_root = get_datasets_path() / "BigBench"
    cot_prompts_dir = bigbench_data_root / "cot-prompts"
    bbh_test_dir = bigbench_data_root / "bbh"
    generations_dir = get_generations_path()

    if not cot_prompts_dir.exists():
        _logger.error(f"COT prompt directory {cot_prompts_dir} does not exist")
        exit(1)
    elif not bbh_test_dir.exists():
        _logger.error(f"BBH test directory {bbh_test_dir} does not exist")
        exit(1)

    _logger.info(f"Processing CoT for BigBench subjects: {subjects}")

    threads = []
    for subject in subjects:
        bbh_test_path = bbh_test_dir / f"{subject}.json"
        cot_prompt_path = cot_prompts_dir / f"{subject}.txt"
        if not bbh_test_path.exists():
            _logger.error(f"Data file {bbh_test_path} does not exist")
            exit(1)
        elif not cot_prompt_path.exists():
            _logger.error(f"COT prompt file {cot_prompt_path} does not exist")
            exit(1)

        if api_type == "completion":
            _logger.info(f"Starting completion thread for {bbh_test_path}")
            results_path = generations_dir / "bigbench" / "cot_results" / "completion"
            if overwrite:
                cot_results_filename = results_path / f"{subject}_completion_cot_results.json"
                if cot_results_filename.exists():
                    cot_results_filename.unlink()
            results_path.mkdir(parents=True, exist_ok=True)
            thread = threading.Thread(
                target=do_completion_cot,
                args=(bbh_test_path, cot_prompt_path, subject, results_path),
            )
        else:
            _logger.info(f"Starting chat thread for {bbh_test_path}")
            results_path = generations_dir / "bigbench" / "cot_results" / "chat"
            results_path.mkdir(parents=True, exist_ok=True)
            if overwrite:
                cot_results_filename = results_path / f"{subject}_chat_cot_results.json"
                if cot_results_filename.exists():
                    cot_results_filename.unlink()
            thread = threading.Thread(
                target=do_chat_cot,
                args=(bbh_test_path, cot_prompt_path, subject, results_path),
            )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("Done!")
