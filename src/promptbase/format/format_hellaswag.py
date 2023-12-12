import json
import os
import csv
import uuid


train_path = "../datasets/hellaswag_train.jsonl"
test_path = "../datasets/hellaswag_test.jsonl"
val_path = "../datasets/hellaswag_val.jsonl"


def process_jsonl_file(file_path, split_name):
    questions = []
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for i, json_line in enumerate(lines):
            question_data = json.loads(json_line)
            answer_choices = {
                chr(65 + i): answer for i, answer in enumerate(question_data["endings"])
            }

            question_dict = {
                "question_number": f"{question_data['ind']}",
                "question": question_data["ctx"],
                "correct_answer": chr(65 + question_data["label"]),
                "has_media": False,  # Assuming no media in MMLU dataset
                "dataset": "hellaswag",
                "id": f"{uuid.uuid4()}",
                "split": split_name,
                "extra": question_data[
                    "activity_label"
                ],  # Any extra information, if needed
                "answer_choices": answer_choices,
            }
            questions.append(question_dict)
    return questions


train_questions = process_jsonl_file(train_path, "train")
# test_questions = process_jsonl_file(test_path, "test")
val_questions = process_jsonl_file(val_path, "val")

print("Train questions: ", len(train_questions))
# print("Test questions: ", len(test_questions))
print("Val questions: ", len(val_questions))

# all_questions = train_questions + test_questions + val_questions
all_questions = train_questions + val_questions

with open("hellaswag.json", "w", encoding="utf-8") as json_file:
    json.dump(all_questions, json_file, ensure_ascii=False, indent=4)
