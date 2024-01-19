import logging
import sys
import textwrap

from typing import Any, Dict

import guidance
from guidance import gen, select, system, user, assistant


_logger = logging.getLogger(__file__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler(stream=sys.stdout))


ANSWER_KEY = "string_choice"
COT_KEY = "explanation"

# Ought to write a generator for this....
PLAIN_HUNT = [
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [1, 3, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 1, 0],
    [2, 3, 0, 1],
    [2, 0, 3, 1],
    [0, 2, 1, 3],
]

NUM_PERMUTATIONS = 5


@guidance
def few_shot_cot_multiple_choice(
    lm: guidance.models.Chat,
    question: str,
    choices: list[str],
    fewshot_examples: list[dict[str, any]],
    permutation: list[int],
):
    # Some general instruction to the model
    with system():
        lm += textwrap.dedent(
            """Answer the following multiple choice **Question**.
            First, think step by step and write an **Explanation** for reasoning through the question.
            Then, when prompted by the user for a **Final Answer**, analyze your explanation and write just the number of the correct answer.
            Do not say the final answer until the user asks for it."""
        )

    for example in fewshot_examples:
        with user():
            lm += "**Question**\n"
            lm += example["question"] + "\n"
            for i, choice in enumerate(example["choices"]):
                lm += f"{i} : {choice}\n"
            lm += "**Explanation**"

        with assistant():
            lm += example["chain_of_thought"]

        with user():
            lm += f"**Final Answer**"

        with assistant():
            lm += str(example["correct_answer"])

    with user():
        lm += question + "\n"
        for i in range(len(choices)):
            lm += f"{i}: {choices[permutation[i]]}"
        lm += "**Explanation**"

    with assistant():
        lm += gen(name=COT_KEY)

    with user():
        lm += f"**Final Answer**"

    with assistant():
        lm += select([str(i) for i in range(len(choices))], name=ANSWER_KEY)

    return lm


def guidance_generation(
    lm: guidance.models.Chat,
    input: Dict[str, Any],
    common: list[dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    _logger.debug("Starting guidance_generation")
    assert common is None, "Unexpected common data"
    assert len(input["choices"]) == 4

    votes = [0, 0, 0, 0]
    cots = []
    for i in range(NUM_PERMUTATIONS):
        current_permutation = PLAIN_HUNT[i]
        result = lm + few_shot_cot_multiple_choice(
            question=input["question"],
            choices=input["choices"],
            fewshot_examples=input["fewshot_examples"],
            permutation=current_permutation,
        )
        _logger.debug(f"Result: {result}")
        cots.append[result[COT_KEY]]
        selected = int(result[ANSWER_KEY])
        actual = current_permutation[selected]
        votes[actual] += 1

    _logger.debug(f"Votes: {votes}")
    # Check the votes
    max_idx = -1
    curr_max = 0
    for i in range(len(votes)):
        if votes[i] > curr_max:
            curr_max = votes[i]
            max_idx = i

    final_result = dict(fewshot_choice=max_idx, fewshot_cot=cots)
    _logger.debug(f"final_result: {final_result}")
    return final_result
