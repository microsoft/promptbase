import json
import re
import sys
from .problem_utils import *

cot_rows1 = load_problems("mmlu/expt/final/MMLU_test/cot_without_rank_knn_5_v0/result")
cot_rows2 = load_problems(
    "mmlu/expt/final/MMLU_test/cot_without_rank_knn_5_gpt-4-1106-preview/result"
)
cot_rows3 = load_problems(
    "mmlu/expt/final/MMLU_test/cot_without_rank_knn_5_gpt-4-1106-preview/result"
)
logprobs_rows1 = load_problems("mmlu/expt/final/MMLU_test/logprobs5_MMLU_dev/result")
logprobs_rows2 = load_problems("mmlu/expt/final/MMLU_test/logprobs5_MMLU_test/result")

import copy


def merge_ds(dataset_list):
    cot_rows = {}
    for rows_set in dataset_list:
        for row in rows_set:
            if row["question_number"] not in cot_rows:
                cot_rows[row["question_number"]] = copy.deepcopy(row)
            else:
                cot_rows[row["question_number"]]["expt"].update(row["expt"])
    return list(cot_rows.values())


cot_rows = merge_ds([cot_rows1, cot_rows2, cot_rows3])
logprobs_rows = merge_ds([logprobs_rows1, logprobs_rows2])

rows = {}
for row in cot_rows:
    key = row["question_number"]
    if key not in rows:
        rows[key] = {}
    rows[key]["question"] = row["question"]
    rows[key]["subject"] = row["extra"].replace("_test", "")
    rows[key]["answer"] = row["correct_answer"]
    expts = row["expt"]
    rows[key]["cot"] = [
        expts[expt]["answer"]
        for expt in expts
        if expts[expt].get("answer", None) is not None
    ]

for row in logprobs_rows:
    key = row["question_number"]
    if key not in rows:
        rows[key] = {}
    rows[key]["question"] = row["question"]
    rows[key]["answer"] = row["correct_answer"]
    expts = row["expt"]
    rows[key]["logprobs"] = [
        expts[expt]["scores"]
        for expt in expts
        if expts[expt].get("scores", None) is not None
    ]

rows = list(rows.values())

data = {}
for row in rows:
    if row["subject"] not in data:
        data[row["subject"]] = []

    scores_logprob = {}
    for e in row["logprobs"]:
        for k in e:
            scores_logprob[k] = scores_logprob.get(k, 0) + e[k] / len(row["logprobs"])

    scores_cot = Counter(row["cot"])
    for k in scores_cot:
        scores_cot[k] /= len(row["cot"])

    data[row["subject"]].append(
        {"logprob": scores_logprob, "cot": scores_cot, "answer": row["answer"]}
    )

# save data to summary.json
with open("summary.json", "w") as f:
    json.dump(data, f, indent=4)
