import concurrent
import concurrent.futures
import datetime
import json
import logging
import os
import random
import re
import string
import time
import types

import numpy as np
import requests
from tqdm import tqdm

logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARN"))

logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
)
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("warnings.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


################################################################################
# File-Related
################################################################################
def load_jsonl(file_name):
    # Load a jsonl file as list of objects
    with open(file_name) as f:
        lines = f.readlines()

    return [json.loads(line) for line in lines if line]


def dump_jsonl(objs, file_name):
    # Save a list/dict of objects to jsonl
    # If it is dict, we only save the values

    with open(file_name, "w") as f:
        if type(objs) is dict:
            for key in objs:
                f.write(json.dumps(objs[key]) + "\n")
        else:
            for obj in objs:
                f.write(json.dumps(obj) + "\n")


################################################################################
# String-Related
################################################################################


def now_string():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def random_string(length=6):
    return "".join(random.choices(string.digits + string.ascii_letters, k=length))


################################################################################
# Multithreading
################################################################################


def run_batch_jobs(run_task, tasks, max_thread):
    """
    Run a batch of tasks with cache.
    - run_task: the function to be called
    - tasks: the list of input for the function
    - max_thread: the number of thread we use
    """
    results = []
    max_failures = 10
    observed_failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_thread) as executor, tqdm(
        total=len(tasks)
    ) as pbar:
        futures = [executor.submit(run_task, task) for task in tasks]

        for future in concurrent.futures.as_completed(futures):
            pbar.update(1)
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.exception("Error occurred during run_batch_jobs.")
                observed_failures += 1
                if observed_failures > max_failures:
                    raise

    return results


################################################################################
# OpenAI API Tools
################################################################################

openai_configs = types.SimpleNamespace()

openai_configs.models = {
    "gpt-4-1106-preview": {"endpoint": "azure", "type": "chat"},
    "text-embedding-ada-002": {"endpoint": "openai-embeddings", "type": "embedding"},
}

openai_configs.endpoints = {
    "openai-embeddings": {
        "headers": {"api-key": f"{os.getenv('AZURE_OPENAI_API_KEY')}"},
        "url": os.getenv("AZURE_OPENAI_EMBEDDINGS_URL"),
    },
    "azure": {
        "headers": {"api-key": f"{os.getenv('AZURE_OPENAI_CHAT_API_KEY')}"},
        "url": os.getenv("AZURE_OPENAI_CHAT_ENDPOINT_URL"),
    },
}

openai_configs.busy_message = [
    "temporarily unable to process your request",
    "The server had an error while processing your request. Sorry about that!",
    "Requests to the Creates a completion for the chat message Operation under Azure OpenAI API",
    "Requests to the Completions_Create Operation under Azure OpenAI API version",
    "Rate limit reached for",
    "exceeded call rate limit",
]
openai_configs.filtered_message = None


def embed(text, model_name="text-embedding-ada-002"):
    model_config = openai_configs.models[model_name]
    endpoint = openai_configs.endpoints[model_config["endpoint"]]
    assert model_config["type"] == "embedding"
    header = endpoint["headers"]
    url = endpoint["url"]

    data = {"input": text, "model": model_name}

    response = requests.post(url, headers=header, json=data, timeout=(30, 600))
    if response.status_code > 400:
        print(f"Got bad response from {url}: {response.text}")
        exit()
    data = response.json()["data"]
    if type(text) == str:
        return data[0]["embedding"]
    else:
        sorted_data = sorted(data, key=lambda x: x["index"])
        sorted_embeddings = [item["embedding"] for item in sorted_data]
        return sorted_embeddings


def embed_batch(texts, model_name="text-embedding-ada-002", batch_size=100):
    embeddings = []
    # Split the input list into sublists of size batch_size
    batches = np.array_split(
        texts,
        len(texts) // batch_size
        if len(texts) % batch_size == 0
        else len(texts) // batch_size + 1,
    )
    for batch in tqdm(batches):
        batch = list(
            batch
        )  # Convert batch back to list if it isn't (depends on array_split output)
        batch_embeddings = embed(batch, model_name)
        embeddings.extend(batch_embeddings)
    return embeddings


def text_completion_impl(
    prompt,
    model="gpt-4-1106-preview",
    temperature=0,
    max_tokens=3500,
    top_p=1.0,
    logprobs=10,
    stop="<|diff_marker|>",
    echo=False,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    max_trial=100,
    **kwargs,
):
    """
    Performs text completion using the openai API with
    - prompt (str or array of str)
    - model ("text-davinci-003", "text-davinci-002", ...)
    - tempature (0 for picking the best token, 1 for more creative solution)
    - max_tokens (limit the total number of generated tokens. 8193 is the maximum context length text-alpha-002)
    - max_trial (the number of retry after getting rate limited warning, we rethrow for all other errors)
    - logprobs (return a list of the most likely tokens and its probabilites. either integer in [1,5] or None)
    - stop (string or list of string (up to 4 strings). The returned text will not contain the stop sequence.)
    """
    last_response = ""
    model_config = openai_configs.models[model]
    endpoint = openai_configs.endpoints[model_config["endpoint"]]
    s = requests.Session()
    if model_config["type"] == "chat":
        if type(prompt) is list and type(prompt[0]) is str:
            assert len(prompt) == 1  # chat model only support 1 prompt at a time
            prompt = prompt[0]
        if type(prompt) is str:
            prompt = [{"role": "user", "content": prompt}]
    elif model_config["type"] == "completion":
        if type(prompt) is list and type(prompt[0]) is dict:
            prompt = (
                "".join(
                    f"<|im_start|>{message['role']}<|im_sep|>{message['content']}<|diff_marker|>"
                    for message in prompt
                )
                + "<|im_start|>assistant"
            )
    else:
        raise "Not supported"

    for retry in range(max_trial):
        last_status_code = 0
        try:
            time.sleep(random.uniform(0, 0.2))
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n": 1,
                "stream": False,
                "top_p": top_p,
                "logprobs": logprobs,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "echo": echo,
                "stop": stop,
            }
            headers = endpoint["headers"]
            if callable(headers):
                headers = headers()

            url = endpoint["url"]

            if model_config["type"] == "chat":
                payload["messages"] = payload["prompt"]
                del payload["prompt"], payload["logprobs"], payload["echo"]

            logging.info("Request:" + str(payload))
            r = s.post(url, headers=headers, json=payload, timeout=200)

            last_response = r.text
            last_status_code = r.status_code
            logging.info(f"{last_status_code} Response:\n" + last_response)
            if (
                r.status_code == 400
                and "The response was filtered due to the prompt triggering Azure OpenAI"
                in r.text
                and openai_configs.filtered_message is not None
            ):
                row = json.loads(r.content)
                row["text"] = openai_configs.filtered_message
                row["finish_reason"] = "content_filter"
                return {"response": {"choices": [row]}, "success": False}

            if r.status_code == 200:
                response = json.loads(r.content)

                for k in range(len(response["choices"])):
                    if response["choices"][k]["finish_reason"] == "content_filter":
                        response["choices"][k]["text"] = openai_configs.filtered_message

                if model_config["type"] == "chat":
                    for k in range(len(response["choices"])):
                        if "message" in response["choices"][k]:
                            response["choices"][k]["text"] = response["choices"][k][
                                "message"
                            ]["content"]
                            del response["choices"][k]["message"]

                if len(response["choices"]) == 1:
                    text = response["choices"][0]["text"]
                else:
                    text = []
                    for r in response["choices"]:
                        text.append(r["text"])

                return {"response": response, "text": text, "success": True}
        except Exception as e:
            logging.exception("Error occurred during HTTP calls in text_completion.")

        filtered_warning = False
        for msg in openai_configs.busy_message:
            if msg in last_response:
                filtered_warning = True

        if not filtered_warning and last_status_code != 429:
            logging.warning(f"{last_status_code} Response:\n" + last_response)

        if last_status_code not in [429, 500, 502, 503, 424]:
            break
    return {"error": last_response, "success": False}


def text_completion(**kwargs):
    result = text_completion_impl(**kwargs)
    if "log_file" in kwargs:
        message = "########## Prompt ##########\n"
        message += (
            str(kwargs["prompt"])
            + "\nmax_tokens="
            + str(kwargs.get("max_tokens", 0))
            + "\n"
        )
        message += "########## Response ##########\n"
        message += result.get("text", "NONE") + "\n"
        with open(kwargs["log_file"], "a") as f:
            f.write(message)
    return result
