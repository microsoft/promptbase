import logging
import sys
import textwrap

from typing import Any, Dict

import guidance
from guidance import gen, select, system, user, assistant


_logger = logging.getLogger(__file__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler(stream=sys.stdout))


@guidance
def few_shot_multiple_choice(
    lm: guidance.models.Chat,
    question: str,
    choices: list[str],
    fewshot_examples: list[dict[str, any]],
):
    # Some general instruction to the model
    with system():
        lm += textwrap.dedent(
            """You are a student taking a multiple choice test.
            You will be shown a question, followed by numbered multiple choice answers.
            Response with the number corresponding to the best answer.
            """
        )

    for example in fewshot_examples:
        with user():
            lm += example["question"] + "\n"
            for i, choice in enumerate(example["choices"]):
                lm += f"{i} : {choice}\n"
            lm += f"Correct Answer: "

        with assistant():
            lm += str(example["correct_answer"])

    with user():
        lm += question + "\n"
        for i, choice in enumerate(choices):
            lm += f"{i} : {choice}\n"
        lm += "Correct Answer: "

    with assistant():
        lm += select([str(i) for i in range(len(choices))], name="string_choice")

    return lm


def guidance_generation(
    lm: guidance.models.Chat,
    input: Dict[str, Any],
    common: list[dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    _logger.debug("Starting guidance_generation")
    assert common is None, "Unexpected common data"
    result = lm + few_shot_multiple_choice(
        question=input["question"],
        choices=input["choices"],
        fewshot_examples=input["fewshot_examples"],
    )

    _logger.debug(f"Result: {result}")

    result = dict(fewshot_choice=int(result["string_choice"]))
    return result
