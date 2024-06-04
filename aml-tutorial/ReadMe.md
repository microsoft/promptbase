# AzureML Tutorial

This directory holds the code for the introduction to AzureML presented at MLADS 2024.
The goal is to provide a foundation for understanding the rather more complex pipelines found in the [`azureml/`](https://github.com/microsoft/promptbase/tree/main/azureml) directory of this repository.

The tutorial implements a simple two-component pipeline, which runs a zero-shot prompt on an MMLU dataset.
The first component runs the dataset line-by-line through a [`guidance`](https://github.com/guidance-ai/guidance) program, storing the model result as an extra key on each line.
The second component compares the answer from the `guidance` program with the correct answer, and computes and overall accuracy score (and a couple of other metrics).

## Prerequisites

To use this tutorial, you need:

- An AzureML workspace
- A chat-enabled OpenAI endpoint, deployed through Azure OpenAI

The AzureML workspace must contain a compute cluster which has an Entra ID authorised to call the Azure OpenAI endpoint.
The tutorial itself allows the use of two compute clusters, one dedicated to the Azure OpenAI endpoint, the other for general use (a pattern taken from the code in `azureml`)

## Setup

After checking out this repository, there are some setup steps to be followed.

First, from the AzureML portal, download the `config.json` file for your workspace.
This contains the subscription, resource group and name of your workspace.
It is used to create the [`MLClient` object](https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.mlclient?view=azure-python) used to interact with AzureML.

Second, create an `other_configs.json` file, based on `other_configs_example.json`.
This is as follows:
```json
{
    "aoai_endpoint": "https://SOMETHING.openai.azure.com/",
    "aoai_deployment": "A_DEPLOYMENT_NAME",
    "aoai_model": "gpt-4-32k",
    "aoai_api_version": "2024-02-01",
    "aoai_compute": "cluster_with_endpoint_permission",
    "general_compute": "any_other_cluster"
}
```
As you can see, this contains information about the Azure OpenAI endpoint, and the compute clusters available for use.
If you only have one, set both the `_compute` keys to the same value.

Finally, install the prerequisites for interacting with AzureML via:
```bash
pip install -r ./requirements.txt
```
then ensure you are logged into Azure (e.g. with `az login`).

## A Note on AzureML versioning

The scripts in this repository always upload new versions of AzureML entities, with the `version` set to the current epoch seconds.
This happens whether or not the underlying code actually changed.
We do this to ensure that the latest code is always being used; trying to figure out exactly what needs to change is error-prone and the size of the uploaded entities is negligible.
In a real system, once things like components have been debugged, we would put them in an AzureML Registry, with more stable versioning.

## Getting the Data

The `create_dataset.py` script will:
- Fetch the specified MMLU dataset
- Reformat it into JSONL
- Save it to AzureML as a Dataset

The script is run with:
```bash
python ./create_dataset.py --workspace_config /path/to/config.json --mmlu_dataset <DATASET NAME> --split <train|test|alidation>
```
The `workspace_config` is the `config.json` file mentioned in the Setup section above.
There are a variety of MMLU datasets, which we [obtain from Hugging Face](https://huggingface.co/datasets/tasksource/mmlu).
These are pre-split into training, test and validation files (listed in descending size).
For the tutorial, you need to pick one (the code in `azureml` will use one split as the questions to be answered, and another for few-shot examples).

Once the script completes successfully, you should see a dataset named `mmlu_<DATASET NAME>_<SPLIT>` in the AzureML portal.
You will need this name to create the pipeline.

## Running the pipeline

The `run_experiment.py` script does the following:
- Creates an AzureML environment
- Creates the two components (which use the environment)
- Uploads the `guidance` program
- Creates the pipeline, and submits it for execution

To run the script:
```bash
python ./run_experiment.py --workspace_config /path/to/config.json --other_config /path/to/other_config.json --dataset_name mmlu_<DATASET NAME>_<SPLIT> --guidance_program ./guidance_programs/zero_shot.py
```
Once this script completes successfully, you should see an experiment in the AzureML portal with a name like `simple_mmlu_<DATASET NAME>_<SPLIT>`, with a freshly created Pipeline run.
If you run the script again, then a new run will be created within the same experiment.

## Comparison to the `azureml/` Code

The code in this directory has been written for simplicity, leading to a number of differences to the [`azureml/`](https://github.com/microsoft/promptbase/tree/main/azureml) directory.
The biggest single difference is that scripts in `azureml/` make use of [the Hydra package](https://hydra.cc/) for configuration management.
This is far more flexible and powerful than the couple of JSON files used here, but its complexity would obscure the AzureML portion of the code.

A number of the Pipelines in `azureml/` make use of sub-pipelines.
These further increase code flexibility and re-use, but are not needed for a basic tutorial.

Finally, the `azureml/` code is more configurable.
The code in this tutorial hard-codes several constants which are configurable in `azureml/`.
Additionally, `azureml/` exposes the arguments which control the encoding used when the JSONL files are saved and loaded.