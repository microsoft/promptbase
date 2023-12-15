import argparse
from promptbase.gsm8k import gsm8k
from promptbase.humaneval import humaneval
from promptbase.math import math
from promptbase.drop import drop
from promptbase.bigbench import bigbench

VALID_DATASETS = ["gsm8k", "humaneval", "math", "drop", "bigbench"]


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument(
        "dataset", type=str, choices=VALID_DATASETS, help="Name of dataset to test"
    )

    return p.parse_args()


def main():
    args = parse_arguments()
    if args.dataset == "gsm8k":
        gsm8k.generate()
        gsm8k.evaluate()
    elif args.dataset == "humaneval":
        humaneval.generate()
        humaneval.evaluate()
    elif args.dataset == "math":
        math.generate()
        math.evaluate()
    elif args.dataset == "drop":
        drop.generate()
        drop.evaluate()
    elif args.dataset == "bigbench":
        bigbench.generate()
        bigbench.evaluate()
    else:
        raise ValueError(f"Bad dataset: {args.dataset}")


if __name__ == "__main__":
    main()
