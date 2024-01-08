import argparse
import gzip
import json
import pathlib
import os

from collections import defaultdict

API_DATA_KEYS = ["api_calls", "tokens_used_prompt", "tokens_used_completion"]


def load_answers(file):
    with open(file, "r") as f:
        return json.load(f)


def load_questions(file_path):
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


def evaluate(user_answers, reference_answers):
    results = defaultdict(lambda: defaultdict(int))

    # Create a dictionary to count total questions for each dataset
    total_questions = defaultdict(int)

    # Create a dictionary to match answer id and its dataset, correct answer
    answer_dict = {}
    for x in reference_answers:
        if x["dataset"] == "MMLU":
            x["dataset"] = (
                x["dataset"] + "_" + "_".join(x["question_number"].split("_")[0:-1])
            )
        answer_dict[x["id"]] = x["correct_answer"]
        total_questions[x["dataset"]] += 1

    for answer in user_answers:
        if answer["id"] in answer_dict:
            dataset = [
                q["dataset"] for q in reference_answers if q["id"] == answer["id"]
            ][0]
            results[dataset]["total"] += 1
            for key in API_DATA_KEYS:
                if results[dataset][key] != "N/A" and key in answer:
                    results[dataset][key] += answer[key]
                else:
                    results[dataset][
                        key
                    ] = "N/A"  # Pretty printing -- if API info doesn't exist, set it to N/A
            if answer["answer"] == answer_dict[answer["id"]]:
                results[dataset]["correct"] += 1

    for dataset, data in results.items():
        accuracy = (data["correct"] / data["total"]) * 100
        print(
            f'{dataset}: ({data["total"]}/{total_questions[dataset]}) Evaluated | ({data["correct"]}/{data["total"]}) Correct ({accuracy:.02f}%)'
        )

    print(f"All results: {json.dumps(results, indent=4)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "user_answers_file",
        type=pathlib.Path,
        help="The JSON file containing user answers",
    )
    parser.add_argument(
        "--do_upload",
        action="store_true",
        help="If present, attempt to upload results to the leaderboard",
    )
    parser.add_argument("--approach", type=str, required=False, default="")
    parser.add_argument("--description", type=str, required=False, default="")
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="The model used to produce results. REQUIRED for upload",
    )
    parser.add_argument(
        "--contact",
        type=str,
        required=False,
        help="Who is uploading the result. REQUIRED for upload",
    )
    args = parser.parse_args()

    print(f"do_upload: {args.do_upload}")
    if args.do_upload:
        assert len(args.model) > 0, "Must specify a model"
        assert len(args.contact) > 0, "Must specify a contact"

    upload_info = dict(
        contact=args.contact,
        approach=args.approach,
        description=args.description,
        model=args.model,
    )
    print(f"upload_info: {upload_info}")

    user_answers = load_answers(args.user_answers_file)

    # TODO: Load dataset using a common path
    answer_path = os.path.join("repo_root", "data", "all_questions_train.json")
    reference_answers = load_questions(answer_path)
    results = evaluate(user_answers, reference_answers)

    print("Complete")
