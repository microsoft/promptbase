import datetime
import json
import os
import pathlib

from promptbase.utils.helpers import get_datasets_path, get_generations_path, get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)

def score(api_type="chat"):
    ground_truth_dir = get_datasets_path() / "BigBench" / "bbh"
    if not ground_truth_dir.exists():
        _logger.error(f"Ground truth directory {ground_truth_dir} does not exist")
        return
    answer_dir = get_generations_path() / "bigbench" / "answers" / api_type

    score_dict = {}

    # loop through json files in ground truth path
    for gt_filename in os.listdir(ground_truth_dir):
        if not gt_filename.endswith(".json"):
            _logger.warn("Skipping non-json file: " + gt_filename)
            continue
        _logger.info("Processing file: " + gt_filename)
        fname_base = gt_filename.split(".")[0]
        answer_path = answer_dir / f"{fname_base}_{api_type}_answers.json"
        if not os.path.exists(answer_path):
            _logger.warn("Answer file does not exist: %s", answer_path)
            continue
        with open(ground_truth_dir / gt_filename) as f:
            ground_truth_data = json.load(f)
        with open(answer_path) as f:
            answer_data = json.load(f)

        _logger.info("Number of ground truth examples: %s", str(len(ground_truth_data["examples"])))
        _logger.info("Number of answer examples: %s", str(len(answer_data)))
        if len(ground_truth_data["examples"]) != len(answer_data):
            _logger.warn("Number of examples does not match for file: %s", gt_filename)
            continue

        correct_count = 0
        total_count = len(ground_truth_data["examples"])

        for i, gt in enumerate(ground_truth_data["examples"]):
            if gt["target"] == answer_data[i]["completion"]:
                correct_count += 1

        score_dict[fname_base] = {
            "correct": correct_count,
            "total": total_count,
            "score": correct_count / total_count,
        }

    total_correct = 0
    total_overall = 0
    for k, v in score_dict.items():
        total_correct += v["correct"]
        total_overall += v["total"]

    score_dict["overall"] = {
        "correct": total_correct,
        "total": total_overall,
        "score": total_correct / total_overall,
    }

    print("Final scores:", score_dict)

    # save as json file
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    score_dir = get_generations_path() / "bigbench" / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    with open(score_dir / f"bigbench_scores_{api_type}_{timestamp}.json", "w") as f:
        json.dump(score_dict, f)
