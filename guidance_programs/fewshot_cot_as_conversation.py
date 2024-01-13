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


@guidance
def few_shot_cot_multiple_choice(
    lm: guidance.models.Chat,
    question: str,
    choices: list[str],
    fewshot_examples: list[dict[str, any]],
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
        for i, choice in enumerate(choices):
            lm += f"{i} : {choice}\n"
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
    result = lm + few_shot_cot_multiple_choice(
        question=input["question"],
        choices=input["choices"],
        fewshot_examples=input["fewshot_examples"],
    )

    _logger.debug(f"Result: {result}")

    result = dict(fewshot_choice=int(result[ANSWER_KEY]), fewshot_cot=result[COT_KEY])
    return result
