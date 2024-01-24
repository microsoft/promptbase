from .problem_utils import *

test_problem = "MMLU_test_chemistry"

subjects = (
    """Astronomy
College Biology
College Chemistry
College Mathematics
College Medicine
College Physics
Conceptual Physics
Econometrics
Electrical Engineering
Elementary Mathematics
High School Biology
High School Chemistry
High School Macroeconomics
High School Mathematics
High School Microeconomics
High School Physics
High School Statistics
Machine Learning
Professional Accounting
Professional Medicine""".replace(
        " ", "_"
    )
    .lower()
    .split("\n")
)


# Load problems
cot_rows_list = [
    load_problems(f"expt/{test_problem}/cot_knn/result"),
    load_problems(f"expt/{test_problem}/cot_via_knn/result"),
]


def merge_ds(dataset_list):
    cot_rows = {}
    for rows_set in dataset_list:
        for row in rows_set:
            if row["question_number"] not in cot_rows:
                cot_rows[row["question_number"]] = copy.copy(row)
                cot_rows[row["question_number"]]["expt"] = {}
            if "expt" in row and row["expt"]:
                for key in row["expt"]:
                    cot_rows[row["question_number"]]["expt"][key] = row["expt"][key]
    return list(cot_rows.values())


cot_rows = merge_ds(cot_rows_list)
logprobs_rows = load_problems(f"expt/{test_problem}/logprobs5/result")

if cot_rows:
    print("Number of COT:", len(cot_rows[42]["expt"].keys()))
if logprobs_rows:
    print("Number of logprobs:", len(logprobs_rows[42]["expt"].keys()))

# Merge datasets
rows = {}
for row in cot_rows:
    key = row["question_number"]
    if key not in rows:
        rows[key] = {}
    rows[key]["question"] = row["question"]
    rows[key]["subject"] = row["extra"].replace("_test", "").replace("_dev", "")
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
    rows[key]["subject"] = row["extra"].replace("_test", "").replace("_dev", "")
    rows[key]["answer"] = row["correct_answer"]
    expts = row["expt"]
    rows[key]["logprobs"] = [
        expts[expt]["scores"]
        for expt in expts
        if expts[expt].get("scores", None) is not None
    ]

rows = list(rows.values())

n_correct = 0
for row in rows:
    if "cot" in row:
        x = Counter(row["cot"])
        for k in x:
            x[k] /= len(row["cot"])
    else:
        x = {}

    if "logprobs" in row:
        for e in row["logprobs"]:
            for k in e:
                if k not in x:
                    x[k] = 0
                if row["subject"] in subjects:
                    x[k] += 0.5 * e[k] / len(row["logprobs"])
                else:
                    x[k] += 2.0 * e[k] / len(row["logprobs"])

    if x:
        selected_answer = max(x, key=x.get)
        if row["answer"] == selected_answer:
            n_correct += 1
    else:
        n_correct += 1 / 4

print("Number of questions:", len(rows))
print("Number of correct answers:", n_correct)
print("Accuracy:", n_correct / len(rows))
