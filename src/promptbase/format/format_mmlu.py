import json
import os
import csv
import uuid


train_dir = "datasets/MMLU_all/auxiliary_train"
dev_dir = "datasets/MMLU_all/dev"
test_dir = "datasets/MMLU_all/test"
val_dir = "datasets/MMLU_all/val"


# Function to process a single CSV file and return a list of question dictionaries
def process_csv_file(file_path, split_name):
    questions = []
    with open(file_path, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        for i, row in enumerate(csv_reader):
            question_text, *answers, correct_answer = row
            answer_choices = {chr(65 + i): answer for i, answer in enumerate(answers)}
            test_name = file_path.split("\\")[-1].replace(".csv", "")

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


all_questions = []

# List csv files in train_dir
train_csv_list = os.listdir(train_dir)
for train_csv in train_csv_list:
    train_path = os.path.join(train_dir, train_csv)
    questions = process_csv_file(train_path, "train")
    all_questions.extend(questions)
    print(json.dumps(questions[3], indent=4, ensure_ascii=False))

for dev_csv in os.listdir(dev_dir):
    dev_path = os.path.join(dev_dir, dev_csv)
    questions = process_csv_file(dev_path, "dev")
    all_questions.extend(questions)
    print(json.dumps(questions[3], indent=4, ensure_ascii=False))

for test_csv in os.listdir(test_dir):
    test_path = os.path.join(test_dir, test_csv)
    questions = process_csv_file(test_path, "test")
    all_questions.extend(questions)
    print(json.dumps(questions[3], indent=4, ensure_ascii=False))

for val_csv in os.listdir(val_dir):
    val_path = os.path.join(val_dir, val_csv)
    questions = process_csv_file(val_path, "val")
    all_questions.extend(questions)
    print(json.dumps(questions[3], indent=4, ensure_ascii=False))


with open("all_mmlu_questions.json", "w", encoding="utf-8") as json_file:
    json.dump(all_questions, json_file, ensure_ascii=False, indent=4)
