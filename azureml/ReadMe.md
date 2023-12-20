# AzureML Pipelines

This directory contains [AzureML pipelines](https://learn.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines?view=azureml-api-2) to run various datasets through a given Azure AI endpoints, and assess the results.
The LLM prompting is done using the [`guidance` package](https://github.com/guidance-ai/guidance).
It is provided as an 'extra' and was not used to generated the reported results.

## Contents

- `components`
  This directory contains the Python [components](https://learn.microsoft.com/en-us/azure/machine-learning/concept-component?view=azureml-api-2) which are used in the AzureML pipelines
- `environments`
   This directory contains the definition of the [AzureML environment](https://learn.microsoft.com/en-us/azure/machine-learning/concept-environments?view=azureml-api-2) shared by the various components
- `pipelines`
   This directory contains the code required to submit the pipelines
- `requirements.txt`
   A standard `pip` file which will install the necessary packages for the pipeline submission to work

Furthermore, the actual `guidance` programs are in the top level `guidance_programs` directory in this repository.

## Preparing to submit a pipeline

In order to submit a pipeline, you will need to give various pieces of information to the submission script (e.g. the AzureML workspace information).
Look in the `pipelines/configs` directory, and you will see a number of `*_template.yaml` files.
You will need to make copies without the '_template' suffix, and fill out the contents.
For exmaple, the `aml_config_template.yaml` needs to be copied to `aml_config.yaml` (in the same directory) and filled out with appropriate information.

## Submitting a pipeline

The pipeline submission scripts all have names prefixed with `submit_`.
To run one:
```bash
python ./submit_mmlu_zeroshot.py -cn zeroshot_config
```
where `zeroshot_config` means the `zeroshot_config.yaml` file in the `configs` directory.