# This is a very naive guidance program for GSM8K

import json
import logging
import sys

from typing import Any, Dict

import guidance


_logger = logging.getLogger(__file__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler(stream=sys.stdout))


@guidance
def zero_shot_gsm8k(
    lm: guidance.models.Instruct,
    question: str,
    examples: list[dict[str, Any]] | None,
):
    # Some general instruction to the model
    lm += """You are taking a maths test\n\n"""

    # Show the few shots
    for e in examples:
        lm += f"Question: {e['question']}\n"
        lm += f"Reasoning:"
        for t in e["thoughts"]:
            lm += t["step"] + t["result"]
        lm += f"Answer: {e['answer']}"
        lm += "\n"
    
    # Now ask the question
    lm += f"Question: {question}\n"
    lm += f"Reasoning:"
    lm += guidance.gen("reasons")
    lm += f"Answer: " + guidance.gen(name="result_string")

    return lm


def guidance_generation(
    lm: guidance.models.Chat,
    input: Dict[str, Any],
    common: list[dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    _logger.debug("Starting guidance_generation")
    if common:
        raise ValueError("Common Data not supported!")
    
    result = lm + zero_shot_gsm8k(question=input["question"], examples=input["examples"])

    _logger.info(f"result_string: {result['result_string']}")

    float_result = float(result['result_string'])

    result = dict(zero_or_few_shot_answer=float_result)
    return result
