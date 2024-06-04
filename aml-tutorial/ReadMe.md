# AzureML Tutorial

This directory holds the code for the introduction to AzureML presented at MLADS 2024.
The goal is to provide a foundation for understanding the rather more complex pipelines found in the `azureml` directory of this repository.

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
It is used to create the `MLClient` object used to interact with AzureML.

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