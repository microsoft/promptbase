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
    lm += """Taking a maths test. Answer the following question and
    show your working.
"""

    if common:
        _logger.debug("Adding few shot examples")
        raise ValueError("common data not yet supported")

    lm += question

    schema_obj = dict(type="object", properties=dict(answer=dict(type="number")))

    return lm + guidance.json(name="json_result_object", schema=schema_obj)


def guidance_generation(
    lm: guidance.models.Chat,
    input: Dict[str, Any],
    common: list[dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    _logger.debug("Starting guidance_generation")
    result = lm + zero_shot_gsm8k(question=input["question"], common=common)

    _logger.info(f"Result: {result}")
    _logger.info(f"JSON portion: {result['json_result_object']}")

    loaded_obj = json.loads(result["json_result_object"])

    result = dict(zero_or_few_shot_answer=loaded_obj["answer"])
    return result
