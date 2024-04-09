# This is a very naive guidance program for working on the "produce JSON" task
# described by PASTA for the BIASBIOS dataset
# This version is for a completion model

import logging
import json
import sys

from textwrap import dedent
from typing import Any, Dict

import guidance
from guidance import gen


_logger = logging.getLogger(__file__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler(stream=sys.stdout))


@guidance
def zeroshot_biosbias_json(lm: guidance.models.Model, short_biography: str):
    lm += dedent(
        f"""Answer the occupation of {short_biography} and generate the answer as json format.
        Here is an example: {{"name": , "occupation": ,}}. 
        Now generate the answer:
"""
    )
    _logger.info(f"lm: {lm}")
    lm += gen(name="model_answer")

    return lm


def guidance_generation(
    lm: guidance.models.Chat, input: Dict[str, Any], common: Any = None
) -> Dict[str, Any]:
    _logger.debug("Starting guidance_generation")
    if common is not None:
        _logger.warn("Got unexpected 'common' argument")
    result = lm + zeroshot_biosbias_json(short_biography=input["context"])

    result = dict(model_answer=result["model_answer"])
    return result
