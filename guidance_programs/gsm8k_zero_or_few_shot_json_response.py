# This is a very naive guidance program for GSM8K

import json
import logging
import sys

from typing import Any, Dict

from jsonschema import validate

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

    response_schema = dict(
        type="object",
        properties=dict(
            thoughts=dict(
                type="array",
                items=dict(
                    type="object",
                    properties=dict(
                        step=dict(type="string"),
                        calculation=dict(type="string"),
                        result=dict(type="string"),
                    ),
                ),
            ),
            result=dict(type="number"),
        ),
    )

    # Show the few shots
    for e in examples:
        lm += f"Question: {e['question']}\n"

        nxt_obj = dict(result=e["answer"], thoughts=[])
        for t in e["thoughts"]:
            nxt_thought = dict(step=t["step"])
            if "result" in t:
                nxt_thought["calculation"] = t["calculation"]
                nxt_thought["result"] += t["result"]
            nxt_obj["thoughts"].append(nxt_thought)

        validate(nxt_obj, schema=response_schema)
        lm += guidance.library._json._to_compact_json(nxt_obj)
        lm += "\n\n"

    # Now ask the question
    lm += f"Question: {question}\n"
    lm += guidance.json(name="response_json", schema=response_schema)

    return lm


def guidance_generation(
    lm: guidance.models.Chat,
    input: Dict[str, Any],
    common: list[dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    _logger.debug("Starting guidance_generation")
    if common:
        raise ValueError("Common Data not supported!")

    result = lm + zero_shot_gsm8k(
        question=input["question"], examples=input["examples"]
    )

    _logger.info(f"result_string: {result['response_json']}")

    loaded_obj = json.loads(result["response_json"])

    result = dict(
        zero_or_few_shot_answer=loaded_obj["result"],
        zero_or_few_show_thoughts=loaded_obj["thoughts"],
        final_lm=str(result),
    )
    return result
