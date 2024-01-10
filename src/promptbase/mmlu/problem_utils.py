import copy
import gzip
import logging
import math
import pathlib
import random
import statistics
from collections import Counter

import numpy as np
from sklearn.svm import LinearSVC
from torch.nn import functional as F
from tqdm import tqdm

from . import prompt_templates
from .eval import *

from .mmlu_paths import mmlu_data_dir, mmlu_generations_dir
from .utils import *

########################################
# Load Problems
########################################

_logger = logging.getLogger(__name__)

MMLU_DATASETS = ["clinical_knowledge"]


problem_files = {
    "MMLU_test": "../datasets/mmlu_questions_test",
    "MMLU_dev": "../datasets/mmlu_questions_dev",
    "MMLU_test_clincial": "../datasets/mmlu_questions_test_clinical",
    "MMLU_test_medicine": "../datasets/mmlu_questions_test_medicine",
    "MMLU_test_moral": "../datasets/mmlu_questions_test_moral",
    "MMLU_test_law": "../datasets/mmlu_test_law",
    "MMLU_dev_law": "../datasets/mmlu_dev_law",
    "MMLU_test_virology": "../datasets/mmlu_questions_test_virology",
    "MMLU_test_chemistry": "../datasets/mmlu_test_chemistry",
    "MMLU_dev_chemistry": "../datasets/mmlu_dev_chemistry",
    "MMLU_test_math": "../datasets/mmlu_test_math",
    "MMLU_dev_math": "../datasets/mmlu_dev_math",
}

default_order = "ABCDEFGHIJK"


def load_problems(file_name):
    if file_name in problem_files:
        file_name = problem_files[file_name]
    else:
        file_name = str(mmlu_data_dir / file_name)

    return load_json_file(file_name + ".json")


def save_problems(file_name, problems):
    with gzip.open(file_name + ".json.gz", "wt") as f:
        f.write(json.dumps(problems))


def random_order_impl(options):
    default_order = list(options)
    random.shuffle(default_order)
    return "".join(default_order)


def random_order(options=default_order, forbidden_orders=[], use_default_first=False):
    if use_default_first and len(forbidden_orders) == 0:
        return options
    for i in range(10000):
        order = random_order_impl(options)
        if order not in forbidden_orders:
            return order
    raise "Cannot find a new order"


def multiple_random_order(options, k):
    round = math.ceil(k / len(options))
    orders = ""
    for _ in range(round):
        orders += random_order_impl(options)
    return orders[-k:]


def set_order(problem, order=default_order):
    description = problem["question"] + "\n\n"
    choices = problem["answer_choices"]
    iter = 0
    reduced_order = ""
    for key in order:
        if key not in choices:
            continue
        option = choices[key].strip(" \n")
        description += f"{default_order[iter]}. {option}\n"
        reduced_order += key
        iter += 1
    problem["order"] = reduced_order
    problem["description"] = description


def load_solutions(file_name, options):
    _logger.info(f"load_solutions: {file_name}")
    only_correct_solution = options.get("only_correct_solution", True)
    solution_difficulty = options.get("solution_difficulty", "all")
    problems = load_problems(file_name)
    for problem in problems:
        problem["solution"] = []

        count = 0
        correct = 0
        for expt in problem["expt"]:
            if (
                type(problem["expt"][expt]["result"]) is not str
                or len(problem["expt"][expt]["result"]) == 0
            ):
                continue
            count += 1
            if problem["correct_answer"] == problem["expt"][expt]["result"][-1]:
                correct += 1
                continue

        if solution_difficulty == "easy" and correct < count:
            continue

        if solution_difficulty == "hard" and correct == count:
            continue

        for expt in problem["expt"]:
            if (
                type(problem["expt"][expt]["result"]) is not str
                or len(problem["expt"][expt]["result"]) == 0
            ):
                continue
            if (
                only_correct_solution
                and problem["correct_answer"] != problem["expt"][expt]["result"][-1]
            ):
                continue
            question = (
                problem["expt"][expt]["prompt"]
                .split("## Question\n")[-1][: -len("\n## Answer\n")]
                .strip("\n ")
            )
            answer = problem["expt"][expt]["response"].strip("\n ")
            problem["solution"].append({"question": question, "answer": answer})
    return problems


def reorder_question(question_str):
    # Extract the main question and options using regex
    main_question, *options = re.split(r"(\n[A-D]. )", question_str)
    options = [options[i] + options[i + 1] for i in range(0, len(options), 2)]

    # Now options list contain all options unordered
    # Let's sort the options
    sorted_options = sorted(options)

    # Combine main question and sorted options
    return main_question + "".join(sorted_options)


########################################
# Parse response
########################################
def parse_MC(problem, response, answer_type="bracket"):
    # If the text has [A], then the answer is A
    text = response["text"]
    if answer_type == "bracket" and "\nAnswer: " in text:
        text = text.split("\nAnswer: ", maxsplit=1)[1]

    answers = ""
    for letter in default_order:
        if (
            answer_type == "bracket"
            and f"[{letter}]" in text
            and letter in problem["order"]
        ):
            answers += letter
        if answer_type == "plain" and letter in text and letter in problem["order"]:
            answers += letter
        if (
            answer_type == "answer"
            and f"Answer: {letter}" in text
            and letter in problem["order"]
        ):
            answers += letter
        if (
            answer_type == "answer_md"
            and f"## Answer\n{letter}" in text
            and letter in problem["order"]
        ):
            answers += letter

    if len(answers) == 1:
        return answers
    else:
        return None


def parse_order(problem, response):
    allowed_char = default_order[: len(problem["order"])]
    pattern = r"\nAnswer: " + " < ".join(
        [r"\[[" + allowed_char + r"]\]"] * len(problem["order"])
    )
    match = re.search(pattern, response["text"])
    if match:
        return (
            match.group(0)[len("\nAnswer: ") :]
            .replace("[", "")
            .replace("]", "")
            .replace("<", "")
            .findreplace(" ", "")
        )
    else:
        print()
        return None


def parse_decreasing_order(problem, response):
    allowed_char = default_order[: len(problem["order"])]
    pattern = r"\nAnswer: " + " > ".join(
        [r"\[[" + allowed_char + r"]\]"] * len(problem["order"])
    )
    match = re.search(pattern, response["text"])
    if match:
        order = (
            match.group(0)[len("\nAnswer: ") :]
            .replace("[", "")
            .replace("]", "")
            .replace(">", "")
            .replace(" ", "")
        )
        return order[::-1]
    else:
        print(response["text"])
        return None


def parse_decreasing_order2(problem, response):
    allowed_char = default_order[: len(problem["order"])]
    pattern = r"## Ranking All Options From Most Likely to Least Likely\n" + ", ".join(
        [r"[" + allowed_char + r"]"] * len(problem["order"])
    )
    match = re.search(pattern, response["text"])
    if match:
        order = match.group(0).split("\n")[1].replace(",", "").replace(" ", "")
        return order[::-1]
    else:
        print(response["text"])
        return None


def parse_scores(problem, response):
    scores = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"(\w) = (\d+)/10", response["text"])
    }
    if len(scores) == 0:
        return None

    order = "".join(sorted(scores, key=scores.get))
    return order, scores


def parse_probs(problem, response):
    scores = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"(\w) = (\d+)%", response["text"])
    }
    if len(scores) == 0:
        return None

    order = "".join(sorted(scores, key=scores.get))
    return order, scores


def parse_logprobs(problem, response):
    scores_raw = response["response"]["choices"][0]["logprobs"]["top_logprobs"][0]
    scores = {}
    for key in scores_raw:
        if key.strip(" \n") not in problem["order"] or key.strip(" \n") == "":
            continue
        scores[key.strip(" \n")] = scores.get(key.strip(" \n"), 0) + math.exp(
            scores_raw[key]
        )
    Z = sum([v for v in scores.values()])
    if Z < 0.2:
        print(scores_raw)
        return None
    order = "".join(sorted(scores, key=scores.get))
    return order, scores


def parse_response(problem, response, mode, reorder=True):
    if mode == "MC":
        result = parse_MC(problem, response, answer_type="bracket")
    elif mode == "letter":
        result = parse_MC(problem, response, answer_type="plain")
    elif mode == "answer_letter":
        result = parse_MC(problem, response, answer_type="answer")
    elif mode == "answer_letter_md":
        result = parse_MC(problem, response, answer_type="answer_md")
    elif mode == "order":
        result = parse_order(problem, response)
    elif mode == "decreasing_order":
        result = parse_decreasing_order(problem, response)
    elif mode == "decreasing_order2":
        result = parse_decreasing_order2(problem, response)
    elif mode == "scores":
        result = parse_scores(problem, response)
    elif mode == "probs":
        result = parse_probs(problem, response)
    elif mode == "logprobs":
        result = parse_logprobs(problem, response)
    else:
        raise "Unsupported mode"

    if reorder:

        def letter_map(x):
            return problem["order"][default_order.find(x)]

        try:
            if type(result) == tuple and len(result) == 2:
                order = "".join([letter_map(x) for x in result[0]])
                scores = {letter_map(key): result[1][key] for key in result[1]}
                return order, scores
            else:
                order = "".join([letter_map(x) for x in result])
                return order
        except:
            return None
    else:
        return result


########################################
# Parse related functions
########################################


def select_examples(problem, examples, mode, options):
    selected = []
    num_examples = options.get("num_examples", 5)

    if mode == "random":
        if "problems" in examples:
            examples = examples["problems"]

        problems = random.sample(examples, num_examples)
        for problem in problems:
            if "solution" in problem:
                solution = random.choice(problem["solution"])
            elif "question" in problem and "answer" in problem:
                solution = problem
            else:
                raise "Wrong format"

            selected.append(
                {"question": solution["question"], "answer": solution["answer"]}
            )
    elif mode == "knn":
        problem_embedding = options["problem_embedding"]
        examples_tensor = examples["tensor"]
        examples = examples["problems"]

        # Then, compute the cosine similarity for each data tensor with respect to the target
        cosine_similarities = F.cosine_similarity(
            examples_tensor, problem_embedding, dim=1
        )

        # Store the cosine similarity scores along with the corresponding indices
        cosine_similarity_scores = [
            (i, cosine_similarity.item())
            for i, cosine_similarity in enumerate(cosine_similarities)
        ]

        # Remove identical examples
        cosine_similarity_scores = [
            item
            for item in cosine_similarity_scores
            if problem["question"] not in examples[item[0]]["question"]
        ]

        # Sort the scores in descending order
        cosine_similarity_scores.sort(key=lambda x: x[1], reverse=True)

        # Add noise to the scores to get different ordering each time
        noise_level = (
            cosine_similarity_scores[0][1] - cosine_similarity_scores[5][1]
        ) * options.get("noise_multipler", 0)
        cosine_similarity_scores = [
            (item[0], item[1] + random.gauss(0, noise_level))
            for item in cosine_similarity_scores
        ]

        # Sort the scores in descending order
        cosine_similarity_scores.sort(key=lambda x: x[1], reverse=True)

        # Select the top k scores
        top_k_scores = cosine_similarity_scores[:num_examples]
        top_k_scores = top_k_scores[
            ::-1
        ]  # invert the order of the samples so that the most relevant is the most 'recent'

        # Print the top k cosine similarity scores
        for index, score in top_k_scores:
            solution = random.choice(examples[index]["solution"])
            selected.append(
                {"question": solution["question"], "answer": solution["answer"]}
            )
    elif mode.lower() == "svm":
        problem_embedding = options["problem_embedding"]
        examples_tensor = examples["tensor"]
        examples = examples["problems"]
        C = options.get("C", 0.001)

        X = np.concatenate(
            [problem_embedding.cpu().numpy(), examples_tensor.cpu().numpy()]
        )
        y = np.concatenate([np.array([1]), np.array([0] * len(examples_tensor))])

        clf = LinearSVC(
            class_weight="balanced", verbose=False, max_iter=30000, tol=1e-6, C=C
        )
        clf.fit(X, y)
        similarities = clf.decision_function(X)

        # Add noise to the scores to get different ordering each time
        sorted_similarities = np.sort(similarities)[::-1]
        noise_level = (sorted_similarities[0] - sorted_similarities[5]) * options.get(
            "noise_multipler", 0
        )
        similarities += noise_level * np.random.rand(*similarities.shape)

        # Select the top k scores
        indices = np.argsort(-similarities)[: num_examples + 1][1:] - 1
        indices = indices.tolist()[
            ::-1
        ]  # invert the order of the samples so that the most relevant is the most 'recent'

        # Print the top k
        for index in indices:
            solution = random.choice(examples[index]["solution"])
            selected.append(
                {"question": solution["question"], "answer": solution["answer"]}
            )
    else:
        raise "Unsupported method."

    return selected


########################################
# Statistics related functions
########################################
# pick the uniquely most common element in the string of "A,B,A,C" => "A"
# If require_unique, "A,B,A,B" => "None"
def most_common_element(s, require_unique=False):
    if len(s) == 0:
        return None

    lst = [el[-1] for el in s.split(",") if type(el) is str and len(el) >= 1]
    counter = Counter(lst)
    most_common = counter.most_common()
    if len(most_common) == 1:
        return most_common[0][0]
    elif most_common[0][1] == most_common[1][1] and require_unique:
        return None
    else:
        return most_common[0][0]


def same_answer(s):
    if len(s) == 0:
        return None

    lst = [el[-1] for el in s.split(",") if type(el) is str and len(el) >= 1]
    counter = Counter(lst)
    most_common = counter.most_common()
    if most_common[0][1] == len(lst):
        return most_common[0][0]
    else:
        return None


def merge_rankings(s):
    # if all None, return None
    if len(s) == 0:
        return None

    if all(x == "None" for x in s.split(",")):
        return None

    # if not all None, continue with processing
    rank_score = {0: 6, 1: 3, 2: 1}
    scores = {letter: 0 for letter in "ABCDEFGHIJK"}

    for perm in s.split(","):
        if perm == "None":
            continue
        perm = perm.strip(" \n")
        for i, letter in enumerate(perm):
            if len(rank_score) > len(perm) - i - 1:
                scores[letter] += rank_score[len(perm) - i - 1]

    # sort letters by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])

    # extract just the letters in the new order
    sorted_letters = [letter for letter, score in sorted_scores]
    rank = "".join(sorted_letters)

    return rank


def variance_estimator(s, answer):
    if len(s) == 0:
        return 0

    lst = [el[-1] == answer for el in s.split(",") if type(el) is str and len(el) >= 1]
    if len(lst) == 1:
        return 0.25
    else:
        return statistics.variance(lst)


def compute_statistics(
    problems, merge_func=merge_rankings, extract_mode=None, top23=False, merge_only=True
):
    stats = {}
    results = {}
    extracted_problems = []
    selected_options = {}

    variance = 0
    for problem in problems:
        if "expt" not in problem:
            continue

        # compute merged answer
        extra = "@" + problem["extra"]
        answers = [
            problem["expt"][expt]["result"]
            for expt in problem["expt"]
            if expt != "^merged" and problem["expt"][expt]["result"] is not None
        ]
        merged_answer = merge_func(",".join(answers))
        if len(problem["expt"]) > 1 or merge_only:
            problem["expt"]["^merged"] = {"result": merged_answer}
        # compute variance
        variance += variance_estimator(",".join(answers), problem["correct_answer"])

        for expt in problem["expt"]:
            if expt not in stats:
                stats[expt] = {
                    "count": 0,
                    "answer": 0,
                    "correct": 0,
                    "top2": 0,
                    "top3": 0,
                }
                if expt != "^merged":
                    results[expt] = []

            if expt + extra not in stats:
                stats[expt + extra] = {
                    "count": 0,
                    "answer": 0,
                    "correct": 0,
                    "top2": 0,
                    "top3": 0,
                }
                if expt != "^merged":
                    results[expt] = []

            stats[expt]["count"] += 1
            stats[expt + extra]["count"] += 1
            answer = problem["correct_answer"]
            if problem["expt"][expt]["result"] is None:
                continue
            result = (
                "ZZZZZZZZZZZZZZ" + problem["expt"][expt]["result"]
            )  # padding if it only gives one answer
            stats[expt]["answer"] += 1
            stats[expt + extra]["answer"] += 1

            if answer in result[-1]:
                stats[expt]["correct"] += 1
                stats[expt + extra]["correct"] += 1

            if answer in result[-2:]:
                stats[expt]["top2"] += 1
                stats[expt + extra]["top2"] += 1

            if answer in result[-3:]:
                stats[expt]["top3"] += 1
                stats[expt + extra]["top3"] += 1

            if expt != "^merged":
                results[expt].append(problem["expt"][expt])
                results[expt][-1]["id"] = problem["id"]
            elif extract_mode is not None and answer in result[-extract_mode:]:
                extracted_problems.append(copy.deepcopy(problem))
                selected_options[problem["id"]] = result[-extract_mode:]

        if "^merged" in problem["expt"]:
            del problem["expt"]["^merged"]

    summary = ""
    alt_acc = 0
    alt_cnt = 0
    for expt in stats:
        if merge_only and "^merged" not in expt:
            continue

        count = stats[expt]["count"]
        answer = stats[expt]["answer"]
        correct = stats[expt]["correct"]
        top2 = stats[expt]["top2"]
        top3 = stats[expt]["top3"]
        alt_acc += correct / (answer + 1e-12)
        alt_cnt += 1
        summary += f"{expt.replace('^merged@', '').replace('_test','')}\t{count}\t{answer}\t{correct}\t{(correct/(answer+1e-12))*100:.1f}\n"

    for expt in results:
        (mmlu_generations_dir / "expt").mkdir(parents=True, exist_ok=True)
        # os.makedirs(os.path.dirname(f"expt/{expt}.json"), exist_ok=True)
        with open(mmlu_generations_dir / "expt" / f"{expt}.json", "w") as f:
            f.write(json.dumps(results[expt]))

    if extract_mode is not None:
        for problem in extracted_problems:
            sorted_option = "ABCDEFGHI"
            selected_option = selected_options[problem["id"]]
            problem["correct_answer"] = sorted_option[
                selected_option.find(problem["correct_answer"])
            ]
            problem["answer_choices"] = {
                sorted_option[idx]: problem["answer_choices"][key]
                for (idx, key) in enumerate(selected_option)
            }
            del problem["expt"]
        with gzip.open(f"extracted.json.gz", "wt") as f:
            f.write(json.dumps(extracted_problems))

    return summary


def ensemble(
    path, first_methods, second_method, merge_func=merge_rankings, verbose=False
):
    answered = {}
    for first_method in first_methods:
        problems = load_problems(f"{path}/{first_method}/result")

        for problem in problems:
            if "expt" not in problem:
                continue

            answers = [
                problem["expt"][expt]["result"]
                for expt in problem["expt"]
                if expt != "^merged" and problem["expt"][expt]["result"] is not None
            ]
            merged_answer = same_answer(",".join(answers))
            if problem["id"] in answered and answered[problem["id"]] != merged_answer:
                answered[problem["id"]] = None
            else:
                answered[problem["id"]] = merged_answer

    problems = load_problems(f"{path}/{second_method}/result")

    count = 0
    correct = 0
    wrong = 0
    for problem in problems:
        count += 1
        if "expt" not in problem:
            continue

        # compute merged answer
        if problem["id"] in answered and answered[problem["id"]] != None:
            merged_answer = answered[problem["id"]]
            answers = "same_answer by 5shot: " + merged_answer
        else:
            answers = [
                problem["expt"][expt]["result"]
                for expt in problem["expt"]
                if expt != "^merged" and problem["expt"][expt]["result"] is not None
            ]
            merged_answer = merge_func(",".join(answers))

        if problem["correct_answer"] == merged_answer[-1]:
            correct += 1
        else:
            wrong += 1
            if verbose:
                print(f"## Problem {wrong}")
                set_order(problem, "ABCDEFGHIJ")
                print(problem["description"])
                print("GPT: " + "".join(answers))
                print(problem["correct_answer"])
                print("")

    summary = f"# {path}\n"
    summary += f"# methods = {first_methods} / {second_method} \n"
    answer = correct + wrong
    if count != answer:
        summary += f"Answered = {answer} / {count} ({answer/(count+1e-12)*100:.1f} %)\n"
    summary += f"Accuracy = {correct} / {answer} ({correct/(answer+1e-12)*100:.1f} %)\n"
    print(summary)
