import gzip
import json
import pathlib

import sklearn.metrics as skm

from .mmlu_paths import mmlu_data_dir, mmlu_generations_dir

API_DATA_KEYS = ["api_calls", "tokens_used_prompt", "tokens_used_completion"]


def load_json_file(file_path):
    if type(file_path) is str:
        file_path = pathlib.Path(file_path)

    gz_path = file_path.with_suffix(file_path.suffix + ".gz")
    print(f"Looking for: {gz_path}")
    if gz_path.exists():
        print("Found zip file")
        with gzip.open(gz_path, "rt") as f:
            return json.load(f)
    else:
        print("Found regular file")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)


def eval_answers(all_questions) -> dict[str, any]:
    y_true = []
    y_pred = []
    answer_counts = []
    skipped = 0
    for item in all_questions:
        answer_voting = dict()
        for response in item["expt"].values():
            if response["answer"] in answer_voting:
                answer_voting[response["answer"]] += 1
            else:
                answer_voting[response["answer"]] = 1
        best_answer = ""
        best_count = 0
        for k, v in answer_voting.items():
            if v > best_count:
                best_answer = k
        if not best_answer:
            skipped += 1
            continue
        y_true.append(item["correct_answer"])
        answer_counts.append(len(answer_voting))
        y_pred.append(best_answer)

    result = dict()
    result["count"] = len(y_true)
    result["accuracy"] = skm.accuracy_score(y_true, y_pred)
    result["skipped"] = skipped
    result["mean_different_answers"] = sum(answer_counts) / len(answer_counts)

    return result


def evaluate_all(dataset_name: str):
    dev_problem = f"mmlu_{dataset_name}_val"
    test_problem = f"mmlu_{dataset_name}_test"

    print(f"Starting evaluation of {dataset_name}")

    variants = {
        "cot": dev_problem,
        "cot_knn": test_problem,
        "cot_via_knn": test_problem,
    }

    for k, v in variants.items():
        print(f"Evaluating {v}")
        # Note that output we have in the directory appears to be a gzip
        all_generated_data = load_json_file(
            mmlu_generations_dir / "expt" / v / k / "result.json"
        )
        stats = eval_answers(all_generated_data)
        print(f"{json.dumps(stats, indent=4)}")
    print("Evaluations complete")
