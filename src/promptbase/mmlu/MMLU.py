import copy
from liquid import Template
from .experiment import *

letter_5shots = {
    "prompt_name": "letter_5shots",
    "response_type": "letter",
    "prompt": Template(
        """{% for item in examples %}## Question
{{ item.question }}

## Answer
{{ item.answer }}

{% endfor %}## Question
{{ question }}
## Answer
"""
    ),
    "examples": [],
}


def run_letter_5shot(
    problem_name, run_name=None, num_repeat=5, max_thread=20, verbose=True
):
    if run_name is None:
        run_name = f"{problem_name}/letter5"

    options = copy.deepcopy(letter_5shots)
    options["problems"] = problem_name
    examples = load_problems(problem_name)
    for example in examples:
        set_order(example, "ABCD")
        example["solution"] = [
            {
                "question": example["description"],
                "answer": example["correct_answer"],
            }
        ]
    options["examples"] = examples
    options["name"] = f"{problem_name}/letter5"
    options["num_examples"] = 5
    options["options"] = "ABCD"
    options["num_repeat"] = num_repeat
    options["max_thread"] = max_thread
    options["verbose"] = verbose
    run_experiment(options)


"""
Which of the following statements is true of the pupillary light reflex?

A. Its efferent limb is carried in the optic nerve
B. It is mediated by the inferior colliculi in the midbrain
C. It is a consensual reflex
D. Its afferent limb is carried in the oculomotor nerve

Let's explain step by step in one paragraph. Then write down your answer in the format of "Answer: "
"""
cot_without_rank = {
    "prompt_name": "cot_without_rank",
    "response_type": "MC",
    "prompt": Template(
        """{% for item in examples %}## Question
{{ item.question }}

## Answer
{{ item.answer }}

{% endfor %}## Question
{{ question }}
## Answer
"""
    ),
    "examples": [
        {
            "question": """Exits on a highway are numbered consecutively from 1 to 50. The distance from exit 41 to exit 50 is 100 km. If each exit is at least 6 km from the next exit, what is the longest possible distance, in kilometers, between exit 47 and exit 48?

A. 52
B. 51
C. 50
D. 49""",
            "answer": """To maximize the distance between exits 47 and 48, we should minimize the distances of the other intervals. There are eight intervals other than the one between exits 47 and 48. Since each exit is at least 6 km from the next, the minimum distance for these eight intervals (from exit 41 to exit 50) would be \(8 \times 6 = 48\) km. Therefore, the distance between exits 47 and 48 would be \(100 - 48 = 52\) km.

Answer: [A]""",
        },
        {
            "question": """The two principal contractile proteins found in skeletal muscle are:
A. actin and troponin.
B. actin and myosin.
C. troponin and tropomyosin.
D. myosin and tropomyosin.""",
            "answer": """Actin is a thin filament protein, forming a helical structure along with tropomyosin and troponin, which regulate its interaction with myosin. Myosin, on the other hand, is a thick filament protein characterized by its long tail and globular head. The heads of the myosin filaments form cross-bridges by attaching to specific sites on the actin filaments. Through a series of movements fueled by ATP hydrolysis, these cross-bridges pull the actin filaments towards the center of the sarcomere, the basic unit of a muscle fiber, causing the muscle to contract. While troponin and tropomyosin are also important in muscle contraction, they serve more as regulatory proteins rather than primary contractile proteins.

Answer: [B]""",
        },
        {
            "question": """A man was angry at a coworker who had received a promotion. The man believed that the coworker had taken credit for the man's work and had bad- mouthed him to their boss. One day, as the man was leaving the company parking lot in his car, he saw the coworker walking through the lot. On a sudden impulse, the man pushed the accelerator pedal hard and veered toward the coworker with the intention of scaring him. The coworker tried to jump out of the way but slipped and fell and was run over. Although the coworker suffered life-threatening injuries, he survived. In a jurisdiction that follows the common law of homicide, could the man properly be convicted of attempted murder? 

A. No, because the coworker's slip and fall broke the chain of proximate causation.
B. No, because the man lacked the requisite intent.
C. Yes, because the coworker suffered life- threatening injuries.
D. Yes, because the man acted with reckless disregard for human life.""",
            "answer": """In this scenario, the key factors to consider are the man's intent and the nature of his actions under the common law of homicide. Attempted murder requires a specific intent to kill. The man's action of veering his car towards the coworker was impulsive and aimed at scaring, not killing. This suggests a lack of specific intent to commit murder.

Answer: [B]""",
        },
        {
            "question": """Which one of the following is not a vectored interrupt?
    
A. TRAP.
B. RST 7.5.
C. RST 6.5.
D. INTR.""",
            "answer": """Vectored interrupts are a type of interrupt where the address of the interrupt service routine (ISR) is hardwired. That means the processor automatically knows where to go when the interrupt occurs. 

- **TRAP** is a vectored interrupt. It has a fixed location in memory, so when the TRAP interrupt occurs, the processor knows exactly where to go.
- **RST 7.5 and RST 6.5** are also vectored interrupts in the Intel 8085 microprocessor. These interrupts have predefined memory addresses, so the processor can jump directly to these addresses when these interrupts are triggered.
- **INTR**, on the other hand, is a non-vectored interrupt. It does not have a predefined memory address. Instead, when the INTR interrupt occurs, the device has to supply the address of the ISR through an external process, typically involving a bus for the address.

Answer: [D]""",
        },
        {
            "question": """A psychologist who believes in the humanistic perspective would be most likely to agree with which of the following statements?

A. All behavior can be traced to human biological functions.
B. Behavior results from conditioning.
C. People's behavior is primarily a result of free will.
D. Human behavior is a result of conflicting unconscious motives.""",
            "answer": """The humanistic perspective in psychology, pioneered by figures like Carl Rogers and Abraham Maslow, emphasizes the importance of free will, self-actualization, and the innate drive towards personal growth. This approach views humans as inherently good and capable of making choices that shape their lives. Unlike the biological perspective (Option A), which focuses on the biological underpinnings of behavior, or the behaviorist perspective (Option B), which emphasizes conditioning and environmental factors in shaping behavior, the humanistic approach centers on individual choice and self-determination. It also differs significantly from the psychoanalytic perspective (Option D), which attributes behavior to unconscious motives and internal conflicts. Thus, the humanistic view is uniquely focused on free will and personal agency in determining human behavior.

Answer: [C]""",
        },
    ],
}


def generate_solutions_without_rank(
    problem_name, run_name=None, model="gpt-4-1106-preview"
):
    if run_name is None:
        run_name = f"{problem_name}/cot_without_rank"
    options = copy.deepcopy(cot_without_rank)
    options["problems"] = problem_name
    options["name"] = run_name
    options["num_examples"] = 5
    options["options"] = "ABCD"
    options["max_thread"] = 40
    options["num_repeat"] = 5
    options["model"] = model
    options["max_retry"] = 2
    run_experiment(options)


def run_logprobs(
    problem_name,
    run_name=None,
    exercise_name=None,
    num_examples=5,
    num_repeat=5,
    max_thread=20,
    verbose=True,
    model="gpt-4-1106-preview",
):
    options = copy.deepcopy(letter_5shots)
    options["response_type"] = "logprobs"
    options["problems"] = problem_name
    options["model"] = model
    if exercise_name is None:
        exercise_name = problem_name
    examples = load_problems(problem_name)
    for example in examples:
        set_order(example, "ABCD")
        example["solution"] = [
            {
                "question": example["description"],
                "answer": example["correct_answer"],
            }
        ]
    options["examples"] = examples
    options["example_selector"] = "knn"
    if run_name is not None:
        options["name"] = run_name
    elif exercise_name is None:
        options["name"] = f"{problem_name}/logprobs{num_examples}"
    else:
        options["name"] = f"{problem_name}/logprobs{num_examples}_{exercise_name}"
    options["num_examples"] = num_examples
    options["options"] = "ABCD"
    options["num_repeat"] = num_repeat
    options["max_thread"] = max_thread
    options["verbose"] = verbose
    run_experiment(options)


def run_letters(
    problem_name,
    run_name=None,
    num_examples=5,
    num_repeat=5,
    max_thread=20,
    verbose=True,
    model="gpt-4-1106-preview",
):
    options = copy.deepcopy(letter_5shots)
    if run_name is None:
        run_name = f"{problem_name}/letters{num_examples}"

    options["response_type"] = "MC"
    options["problems"] = problem_name
    options["model"] = model
    examples = load_problems(problem_name)
    for example in examples:
        set_order(example, "ABCD")
        example["solution"] = [
            {
                "question": example["description"],
                "answer": example["correct_answer"],
            }
        ]
    options["examples"] = examples
    options["example_selector"] = "knn"
    options["name"] = run_name
    options["num_examples"] = num_examples
    options["options"] = "ABCD"
    options["num_repeat"] = num_repeat
    options["max_thread"] = max_thread
    options["verbose"] = verbose
    run_experiment(options)


def run_cot_without_rank(
    problem_name,
    run_name=None,
    mode="knn",
    examples="expt/final/MMLU_dev/cot_without_rank/result",
    num_examples=20,
    num_repeat=5,
    balance_answer=True,
    max_thread=20,
    verbose=True,
    model="gpt-4-1106-preview",
):
    options = copy.deepcopy(cot_without_rank)
    if run_name is None:
        run_name = f"{problem_name}/cot_without_rank_{mode}_{num_examples}_{model}"

    options["examples"] = examples
    options["example_selector"] = mode
    options["num_examples"] = num_examples
    options["balance_answer"] = balance_answer
    options["name"] = run_name
    options["problems"] = problem_name
    options["options"] = "ABCD"
    options["num_repeat"] = num_repeat
    options["max_thread"] = max_thread
    options["retry"] = 10
    options["verbose"] = verbose
    options["model"] = model
    run_experiment(options)
