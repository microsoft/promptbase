import logging
import sys
import textwrap

from typing import Any, Dict, Iterator, TypeVar

import guidance
from guidance import gen, select, system, user, assistant


_logger = logging.getLogger(__file__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler(stream=sys.stdout))


ANSWER_KEY = "string_choice"
COT_KEY = "explanation"


def validate_and_sort_swaps(swaps: list[int], line_len: int) -> list[int]:
    swap_set = set(swaps)
    assert len(swap_set) == len(swaps), f"Swaps not unique: {swaps}"
    for s in swaps:
        assert s - 1 not in swap_set, f"Swaps too close: {s} {swaps}"
        assert s + 1 not in swap_set, f"Swaps too close: {s} {swaps}"
        assert s >= 0, f"Negative swap: {s}"
        assert s < (line_len - 1), f"Swap too large: {s}"
    return list(sorted(swaps))


T = TypeVar("T")


def apply_swaps(line: list[T], swaps: list[int]) -> list[T]:
    sorted_swaps = validate_and_sort_swaps(swaps, len(line))

    i_swap = 0
    result = []
    for i in range(len(line)):
        if i_swap < len(sorted_swaps) and i == sorted_swaps[i_swap]:
            result.append(line[sorted_swaps[i_swap] + 1])
        elif i_swap < len(sorted_swaps) and i == sorted_swaps[i_swap] + 1:
            result.append(line[sorted_swaps[i_swap]])
            i_swap += 1
        else:
            result.append(line[i])
    return result


def plain_hunt_generator(starting_line: list[T]) -> Iterator[T]:
    first_element = starting_line[0]
    swaps_A = list(range(0, len(starting_line) - (len(starting_line) % 2), 2))
    swaps_B = list(range(1, len(starting_line) - 1, 2))
    all_swaps = [swaps_A, swaps_B]
    current = [x for x in starting_line]
    line_count = 0
    yield current
    while True:
        current = apply_swaps(current, all_swaps[line_count % len(all_swaps)])
        yield current
        line_count += 1
        if current[0] == first_element:
            break


NUM_PERMUTATIONS = 5


@guidance
def few_shot_cot_multiple_choice(
    lm: guidance.models.Chat,
    question: str,
    choices: list[str],
    fewshot_examples: list[dict[str, any]],
    permutation: list[int],
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
        for i in range(len(choices)):
            lm += f"{i}: {choices[permutation[i]]}\n"
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

    num_choices = len(input["choices"])

    votes = [0 for _ in range(num_choices)]
    cots = []
    generator = plain_hunt_generator(list(range(num_choices)))
    for i in range(NUM_PERMUTATIONS):
        current_permutation = next(generator)
        result = lm + few_shot_cot_multiple_choice(
            question=input["question"],
            choices=input["choices"],
            fewshot_examples=input["fewshot_examples"],
            permutation=current_permutation,
        )
        _logger.debug(f"Result: {result}")
        cots.append(result[COT_KEY])
        selected = int(result[ANSWER_KEY])
        actual = current_permutation[selected]
        votes[actual] += 1

    _logger.debug(f"Votes: {votes}")
    # Check the votes
    max_idx = -1
    curr_max = 0
    for i in range(len(votes)):
        if votes[i] > curr_max:
            curr_max = votes[i]
            max_idx = i

    final_result = dict(fewshot_choice=max_idx, fewshot_cot=cots)
    _logger.debug(f"final_result: {final_result}")
    return final_result
