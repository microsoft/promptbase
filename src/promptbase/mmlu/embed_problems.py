import gzip
from tqdm import tqdm
from .eval import *
from .utils import *


def embed_file(file_name):
    ds = load_json_file(file_name)
    questions = [row["question"] for row in ds]
    embeddings = embed_batch(questions)
    for row, embedding in tqdm(zip(ds, embeddings)):
        row["embedding"] = embedding

    with gzip.open(file_name + ".gz", "wt") as f:
        json.dump(ds, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question_file", help="The JSON file containing user answers")
    args = parser.parse_args()

    embed_file(args.question_file)
