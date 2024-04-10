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
    common: list[dict[str, Any]] | None,
):
    # Some general instruction to the model
    lm += """Taking a maths test. Answer the following question. Respond with just the numerical answer:
"""

    if common:
        _logger.debug("Adding few shot examples")
        raise ValueError("common data not yet supported")

    lm += question

    return lm + guidance.gen(name="result_string")


def guidance_generation(
    lm: guidance.models.Chat,
    input: Dict[str, Any],
    common: list[dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    _logger.debug("Starting guidance_generation")
    result = lm + zero_shot_gsm8k(question=input["question"], common=common)

    _logger.info(f"JSON portion: {result['result_string']}")

    float_result = float(result['result_string'])

    result = dict(zero_or_few_shot_answer=float_result)
    return result
