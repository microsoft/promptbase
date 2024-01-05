import traceback
from functools import partial
import re

import torch

from .problem_utils import *
from .utils import openai_configs, run_batch_jobs, text_completion

from .mmlu_paths import mmlu_generations_dir


def prepare_options(options):
    # Set experiment name
    if "name" not in options:
        if type(options["problems"]) is not str or "/" in options["problems"]:
            raise "You need to set the experiment name."
        name = options["problems"] + "/" + options["prompt_name"]

        if options.get("example_selector", "") == "knn":
            name += "_knn"
        if "embedding_map" in options:
            name += "_weighted"
        options["name"] = name

    if (
        mmlu_generations_dir / f"expt/{options['name']}"
    ).exists() and "ignore_check" not in options:
        confirm = input(
            f"Are you sure you want to overwrite the content in {options['name']}? (y/n): "
        )
        if confirm.lower() != "y":
            quit()

    if "log_file" not in options:
        options["log_file"] = str(
            mmlu_generations_dir / f"expt" / f"{options['name']}" / "log.md"
        )
        os.makedirs(os.path.dirname(options["log_file"]), exist_ok=True)
        if os.path.exists(options["log_file"]):
            os.remove(options["log_file"])

    # Load problems
    if type(options["problems"]) is str:
        options["problems_name"] = options["problems"]
        options["problems"] = [options["problems"]]

    if type(options["problems"]) is list and type(options["problems"][0]) is str:
        print("Loading problems...")
        problems = []
        for problem_name in options["problems"]:
            problems += load_problems(problem_name)
        options["problems"] = problems

    # Load embedding
    if (
        type(options.get("embedding_map", "")) is bool
        and options["embedding_map"] == True
    ):
        options["embedding_map"] = torch.load("data/matrix_v7_2048dim.pt").cuda()

    if options.get("example_selector", "") in ["knn", "svm"]:
        options["embedding_map"] = torch.eye(1536).cuda()

    # Load examples
    if "examples" not in options:
        options["examples"] = []

    if type(options["examples"]) is str:
        print("Loading examples...")
        examples = load_solutions(options["examples"], options)
        options["examples"] = [
            example for example in examples if len(example["solution"]) > 0
        ]

    if options.get("example_selector", "") in ["knn", "svm"]:
        examples_tensor = (
            torch.cat(
                [
                    torch.tensor(d["embedding"], dtype=torch.float32).unsqueeze(0)
                    for d in options["examples"]
                ]
            ).cuda()
            @ options["embedding_map"]
        )
        options["examples"] = {
            "problems": options["examples"],
            "tensor": examples_tensor,
        }

    # defaults settings
    if "example_selector" not in options and len(options["examples"]) > 0:
        options["example_selector"] = "random"

    # debug settings
    if "debug" in options and options["debug"]:
        options["max_questions"] = 20
        options["num_repeat"] = 2

    if (
        "max_questions" in options
        and len(options["problems"]) > options["max_questions"]
    ):
        random.shuffle(options["problems"])
        options["problems"] = options["problems"][: options["max_questions"]]


def has_common_k_gram(str1, str2, k):
    def generate_k_grams(s, k):
        if len(s) <= k:
            return {s}
        else:
            return {s[i : i + k] for i in range(len(s) - k)}

    str1_k_grams = generate_k_grams(str1, k)
    str2_k_grams = generate_k_grams(str2, k)

    return not str1_k_grams.isdisjoint(str2_k_grams)


def solve(options, problem):
    prompt_tokens = 0
    completion_tokens = 0
    calls = 0
    name = f"{options['name']}/{options['order']}"
    model = options.get("model", "gpt-4-1106-preview")
    set_order(problem, options["order"])

    for retry in range(options.get("max_retry", 3)):
        if "example_selector" not in options:
            selected_examples = []
        elif options["example_selector"] in ["random", "knn", "svm"]:
            if options["example_selector"] in ["knn", "svm"]:
                options["problem_embedding"] = (
                    torch.tensor(problem["embedding"], dtype=torch.float32)
                    .cuda()
                    .unsqueeze(0)
                    @ options["embedding_map"]
                )
            selected_examples = select_examples(
                problem, options["examples"], options["example_selector"], options
            )
        else:
            raise f"Unsupported example_selector: {options['example_selector']}"

        # just some safety check to avoid leaking the answer
        selected_examples = [
            example
            for example in selected_examples
            if example["question"].strip() != problem["question"].strip()
        ]

        if options.get("balance_answer", False):
            assert options["response_type"] in ["decreasing_order", "MC"]
            target_choices = multiple_random_order(
                problem["order"], len(selected_examples)
            )
            for example, target_choice in zip(selected_examples, target_choices):
                result = parse_response(
                    {"order": problem["order"]},
                    {"text": example["answer"]},
                    options["response_type"],
                    False,
                )
                if result is None:
                    continue
                if type(result) is tuple:
                    result = result[0]
                result = result[-1]
                if result == target_choice:
                    continue

                # This is a really bad code here...I assumed the question format.
                assert (
                    f"\n{target_choice}. " in example["question"]
                    and f"\n{result}. " in example["question"]
                )

                example["question"] = (
                    example["question"]
                    .replace(f"\n{target_choice}. ", f"\nTMP. ")
                    .replace(f"\n{result}. ", f"\n{target_choice}. ")
                    .replace(f"\nTMP. ", f"\n{result}. ")
                )
                example["answer"] = (
                    example["answer"]
                    .replace(f"[{target_choice}]", f"[TMP]")
                    .replace(f"[{result}]", f"[{target_choice}]")
                    .replace(f"[TMP]", f"[{result}]")
                )

                example["question"] = reorder_question(example["question"])

        assessments = {}
        if "assessment" in assessments:
            for letter in problem["order"]:
                option = problem["answer_choices"][letter]
                prompt = options["assessment"].render(
                    question=problem["question"], option=option
                )
                response = text_completion(
                    prompt=prompt,
                    temperature=0,
                    max_tokens=300 + retry * 200,
                    stop=["<|diff_marker|>", "\n#"],
                    log_file=options["log_file"],
                )
                if response["success"]:
                    assessments[letter] = response["text"].strip(" \n")
                    prompt_tokens += response["response"]["usage"]["prompt_tokens"]
                    completion_tokens += response["response"]["usage"][
                        "completion_tokens"
                    ]
                    calls += 1

        prompt = options["prompt"].render(
            question=problem["description"],
            examples=selected_examples,
            assessments=assessments,
        )
        if options["response_type"] in ["logprobs", "letter"]:
            max_tokens = 1
        else:
            max_tokens = 500 + retry * 200

        if openai_configs.models[model]["type"] == "chat":
            prompt_splits = re.split("## Q|## A", prompt)
            messages = []
            for prompt_split in prompt_splits:
                if prompt_split.startswith("uestion"):
                    content = prompt_split[len("uestion") :].strip()
                    messages.append({"role": "user", "content": content})
                elif prompt_split.startswith("nswer"):
                    content = prompt_split[len("nswer") :].strip()
                    if content == "":
                        continue
                    messages.append({"role": "assistant", "content": content})
            if "## Rank the options from most likely to least likely" in prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": "Please answer the above question in the same style as the examples above. Include the answer and the rank as the examples above.",
                    }
                )
            else:
                messages.append(
                    {
                        "role": "system",
                        "content": "Please note the exam question has an unique answer based on the information given. Include Answer: [X] at the end where X must be A, B, C, or D.",
                    }
                )
            response = text_completion(
                model=model,
                prompt=messages,
                temperature=0.05 * retry,
                max_tokens=max_tokens,
                log_file=options["log_file"],
            )
        else:
            response = text_completion(
                model=model,
                prompt=prompt,
                temperature=0.05 * retry,
                max_tokens=max_tokens,
                logprobs=10,
                stop=["<|diff_marker|>", "## Question"],
                log_file=options["log_file"],
            )

        if response["success"]:
            prompt_tokens += response["response"]["usage"]["prompt_tokens"]
            completion_tokens += response["response"]["usage"]["completion_tokens"]
            calls += 1
            result = parse_response(problem, response, options["response_type"])
            if result is not None:
                with open(options["log_file"], "a") as f:
                    f.write(
                        f"########## Answer ##########\n{problem['correct_answer']} (GPT answer: {result})\n"
                    )
                break
            else:
                with open(options["log_file"], "a") as f:
                    f.write(
                        f"########## Invalid Answer ##########\n{problem['correct_answer']} (GPT answer: {response['text']})\n"
                    )

    if "expt" not in problem:
        problem["expt"] = {}

    output = {
        "prompt": prompt,
        "response": response["text"],
        "api_calls": calls,
        "tokens_used_prompt": prompt_tokens,
        "tokens_used_completion": completion_tokens,
    }

    if type(result) == tuple:
        output["result"] = result[0]
        output["scores"] = result[1]
    else:
        output["result"] = result

    if type(output["result"]) is str and len(output["result"]) >= 1:
        output["answer"] = output["result"][-1]
    else:
        output["answer"] = None

    problem["expt"][name] = output


def run_experiment(options):
    prepare_options(options)
    try:
        used_order = ["ACDB", "ADBC", "BCDA", "CBAD", "CDAB"]
        for _ in range(options.get("num_repeat", 5)):
            if options.get("reorder", True):
                order = random_order(options.get("options", "ABCDEFGHIJ"), used_order)
            else:
                order = "ABCDEFGHIJ" + random_order(
                    "KLMNOPQRST", used_order
                )  # TODO: cleanup file naming
            used_order.append(order)
            options["order"] = order
            run_batch_jobs(
                partial(solve, options),
                options["problems"],
                max_thread=options.get("max_thread", 30),
            )

            summary = compute_statistics(options["problems"])
            if options.get("verbose", True):
                print(summary)
            save_problems(
                str(mmlu_generations_dir / f"expt" / f"{options['name']}" / "result"),
                options["problems"],
            )  # save results regularly

        with open("summary.md", "a") as f:
            f.write(summary)
            f.write("\n" + "=" * 80 + "\n")
    except KeyboardInterrupt:
        quit()
    except:
        traceback.print_exc()
