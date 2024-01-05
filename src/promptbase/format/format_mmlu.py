import argparse
import csv
import json
import pathlib
import uuid


ALL_QUESTIONS = "all_questions.json"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mmlu_csv_dir", type=pathlib.Path, required=True)
    parser.add_argument("--output_path", type=pathlib.Path, required=True)

    args = parser.parse_args()

    return args


# Function to process a single CSV file and return a list of question dictionaries
def process_csv_file(file_path: pathlib.Path, split_name: str):
    questions = []
    with open(file_path, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        for i, row in enumerate(csv_reader):
            question_text, *answers, correct_answer = row
            answer_choices = {chr(65 + i): answer for i, answer in enumerate(answers)}
            test_name = file_path.stem

            question_dict = {
                "question_number": f"{test_name}_{i}",
                "question": question_text,
                "correct_answer": correct_answer,
                "has_media": False,  # Assuming no media in MMLU dataset
                "dataset": "MMLU",
                "id": f"{uuid.uuid4()}",
                "split": split_name,
                "extra": test_name,  # Any extra information, if needed
                "answer_choices": answer_choices,
            }
            questions.append(question_dict)
    return questions


def main(mmlu_csv_dir: pathlib.Path, output_path: pathlib.Path):
    assert mmlu_csv_dir.is_dir()
    assert output_path.is_dir()
    all_questions = []

    splits = dict(
        train=mmlu_csv_dir / "auxiliary_train",
        dev=mmlu_csv_dir / "dev",
        test=mmlu_csv_dir / "test",
        val=mmlu_csv_dir / "val",
    )

    for split_name, split_path in splits.items():
        for csv_file in split_path.iterdir():
            questions = process_csv_file(csv_file, split_name)
            print(json.dumps(questions[3], indent=4, ensure_ascii=False))
            with open(
                output_path / f"mmlu_{csv_file.stem}.json",
                "w",
                encoding="utf-8",
            ) as json_file:
                json.dump(questions, json_file, ensure_ascii=False, indent=4)
            all_questions.extend(questions)

    with open(output_path / ALL_QUESTIONS, "w", encoding="utf-8") as json_file:
        json.dump(all_questions, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.mmlu_csv_dir, args.output_path)
