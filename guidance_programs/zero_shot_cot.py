# This is a very naive guidance program for doing zero shot multiple choice questions
# with chain-of-thought prompting
# It is not what generated the reported results

import logging
import sys

from typing import Any, Dict

import guidance
from guidance import gen, select, system, user, assistant


_logger = logging.getLogger(__file__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler(stream=sys.stdout))


@guidance
def zero_shot_cot_multiple_choice(
    lm: guidance.models.Chat, question: str, choices: list[str]
):
    # Some general instruction to the model
    with system():
        lm += """You are a student taking a multiple choice test.
You will be shown a question, and then asked to analyse each of the numbered possible responses.
At the end, you will be asked to chose the best response, based on your analyses"""

    with user():
        lm += question

    for i, choice in enumerate(choices):
        with user():
            lm += "Analyse whether the following response is correct:\n"
            lm += f"{i} : {choice}"
        
        with assistant():
            lm += gen(name=f"cot_{i}")

    response_choices = [str(i) for i in range(len(choices))]

    with user():
        lm += f"Based on the above analyses, give the number ({response_choices}) of the correct response:"

    with assistant():
        lm += select(response_choices, name="string_choice")

    return lm


def guidance_generation(
    lm: guidance.models.Chat, input: Dict[str, Any]
) -> Dict[str, Any]:
    _logger.info("Starting guidance_generation")
    result = lm + zero_shot_cot_multiple_choice(
        question=input["question"], choices=input["choices"]
    )

    result = dict(zeroshot_choice=int(result["string_choice"]))
    return result
