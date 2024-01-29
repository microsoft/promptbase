# This is a very naive guidance program for working on the "produce JSON" task
# described by PASTA for the BIASBIOS dataset

import logging
import json
import sys

from textwrap import dedent
from typing import Any, Dict

import guidance
from guidance import gen, select, system, user, assistant


_logger = logging.getLogger(__file__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler(stream=sys.stdout))


NAME_KEY = "given_name"
OCCUPATION_KEY = "occupation"


@guidance
def zeroshot_biosbias_json(lm: guidance.models.Chat, short_biography: str):
    # Some general instruction to the model
    with system():
        lm += dedent(
            """You will be shown a short biography of a person by the user. Answer their questions"""
        )

    with user():
        lm += short_biography

    with assistant():
        lm += "OK"

    with user():
        lm += f"What is the given name of the person? Only reply with their name and nothing else."

    with assistant():
        lm += gen(name=NAME_KEY)

    with user():
        lm += dedent(
            """Simply state the occupation of the person in lower case.
            For example, if a person were an orthodontist, you should state that they are a dentist.
            If the person were a freighter pilot, you should state that they are a pilot.
            Only reply with their occupation and nothing else."""
        )

    with assistant():
        lm += gen(name=OCCUPATION_KEY)

    return lm


def guidance_generation(
    lm: guidance.models.Chat, input: Dict[str, Any], common: Any = None
) -> Dict[str, Any]:
    _logger.debug("Starting guidance_generation")
    if common is not None:
        _logger.warn("Got unexpected 'common' argument")
    result = lm + zeroshot_biosbias_json(short_biography=input["context"])

    result = dict(name=result[NAME_KEY], occupation=result[OCCUPATION_KEY])
    return dict(model_answer=json.dumps(result))
