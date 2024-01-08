import json, random, copy
import numpy as np
from tqdm import tqdm
from collections import Counter

with open("summary.json") as f:
    data = json.load(f)


def calculate_result(rows):
    best_weight = 0
    best_acc = 0
    for weight in np.arange(0, 2, 0.01):
        n_correct = 0
        n_cnt = 0
        for row in rows:
            x = copy.deepcopy(row["cot"])
            for k in row["logprob"]:
                x[k] = x.get(k, 0) + weight * row["logprob"][k]

            selected_answer = max(x, key=x.get)
            n_cnt += 1
            if row["answer"] == selected_answer:
                n_correct += 1
        acc = n_correct / len(rows)
        if acc > best_acc:
            best_acc = acc
            best_weight = weight
    return best_acc, best_weight


# 89.93
subject_weight = 0.5
non_subject_weight = 1.2
subject_list = []
total_correct = 0
total_count = 0
if 1:
    for subject in data:
        print(subject)
        rows = data[subject]

        # use best threshold to process each row
        for i, row in tqdm(enumerate(rows)):
            rows_i = [item for index, item in enumerate(rows) if index != i]
            acc, weight = calculate_result(rows_i)
            x = row["cot"]

            for k in row["logprob"]:
                x[k] = x.get(k, 0) + weight * row["logprob"][k]
            selected_answer = max(x, key=x.get)
            total_count += 1
            if row["answer"] == selected_answer:
                total_correct += 1

if 0:
    for subject in tqdm(data):
        rows = data[subject]
        subject_acc = calculate_result(rows, subject_weight)
        non_subject_acc = calculate_result(rows, non_subject_weight)
        if subject_acc > non_subject_acc:
            weight = subject_weight
        else:
            weight = non_subject_weight

        # use best threshold to process each row
        for i, row in enumerate(rows):
            x = row["cot"]
            for k in row["logprob"]:
                x[k] = x.get(k, 0) + weight * row["logprob"][k]
            selected_answer = max(x, key=x.get)
            total_count += 1
            if row["answer"] == selected_answer:
                total_correct += 1

print(f"total_correct: {total_correct}")
print(f"total_count: {total_count}")
print(f"accuracy: {total_correct / total_count}")
# save best_thresholds to best_thresholds.json
with open("best_thresholds.json", "w") as f:
    json.dump(subject_list, f, indent=4)
