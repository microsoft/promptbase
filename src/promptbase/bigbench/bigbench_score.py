import datetime
import json
import os
import pathlib

my_path = pathlib.Path(__file__).parent.resolve()


def score(api_type="chat"):
    ground_truth_dir = my_path.parent / "datasets" / "BigBench" / "bbh"
    assert ground_truth_dir.exists(), f"Checking for {ground_truth_dir}"
    assert ground_truth_dir.is_dir()
    answer_dir = my_path / "results" / "answers"

    score_dict = {}

    # loop through json files in ground truth path
    for filename in os.listdir(ground_truth_dir):
        if not filename.endswith(".json"):
            print("Skipping non-json file: " + filename)
            continue
        print("Processing file: " + filename)
        fname_base = filename.split(".")[0]
        answer_path = os.path.join(answer_dir, f"{fname_base}_{api_type}_answers.json")
        if not os.path.exists(answer_path):
            print("Answer file does not exist: " + answer_path)
            continue
        with open(os.path.join(ground_truth_dir, filename)) as f:
            ground_truth_data = json.load(f)
        with open(answer_path) as f:
            answer_data = json.load(f)

        print(
            "Number of ground truth examples: "
            + str(len(ground_truth_data["examples"]))
        )
        print("Number of answer examples: " + str(len(answer_data)))
        if len(ground_truth_data["examples"]) != len(answer_data):
            print("Number of examples does not match for file: " + filename)
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

    print(score_dict)

    # save as json file
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"bigbench_scores_{api_type}_{timestamp}.json", "w") as f:
        json.dump(score_dict, f)
