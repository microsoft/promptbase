# This is a very naive guidance program for doing zero shot multiple choice questions
# It is not what generated the reported results

import logging
import sys

from typing import Any, Dict

import guidance
from guidance import select, system, user, assistant


_logger = logging.getLogger(__file__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler(stream=sys.stdout))


@guidance
def zero_shot_multiple_choice(
    lm: guidance.models.Chat,
    question: str,
    choices: list[str],
    common: list[dict[str, Any]] | None,
):
    # Some general instruction to the model
    with system():
        lm += """You are a student taking a multiple choice test.
You will be shown a question, followed by numbered multiple choice answers.
Response with the number corresponding to the best answer.
"""

        if common:
            _logger.info("Adding few shot examples")
            lm += "\nHere are some examples to help you:\n\n"
            for i, example in enumerate(common):
                lm += f"Example {i}\n"
                lm += example["question"] + "\n"
                for j, choice in enumerate(example["choices"]):
                    lm += f"{j+1} : {choice}\n"
                lm += f"Correct Answer: {example['correct_answer']+1}\n\n"

            lm += "The question you need to answer will be shown next.\n\n"

    with user():
        lm += question + "\n"
        for i, choice in enumerate(choices):
            lm += f"{i+1} : {choice}\n"
        lm += "Correct Answer: "

    with assistant():
        lm += select([str(i + 1) for i in range(len(choices))], name="string_choice")

    return lm


def guidance_generation(
    lm: guidance.models.Chat,
    input: Dict[str, Any],
    common: list[dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    _logger.info("Starting guidance_generation")
    result = lm + zero_shot_multiple_choice(
        question=input["question"], choices=input["choices"], common=common
    )

    _logger.info(f"Result: {result}")
    int_result = int(result["string_choice"])

    result = dict(zero_or_few_shot_choice=int_result - 1)
    return result
