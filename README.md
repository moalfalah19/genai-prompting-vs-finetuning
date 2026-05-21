[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/2yXbP9WR)
# Assignment 3: Prompting vs Fine Tuning

This repository contains code that compares two language model techniques: prompting and fine-tuning.

## Requirements

* Python 3.10+
* PyTorch
* Ollama
* Unsloth
* See the full list of dependencies in ```pyproject.toml```

## Task Description

In this project, the goal is to solve a fact-checking classification problem using Large Language Models (LLMs). Given a claim and supporting evidence/document context, the model must predict the correct verdict label (i.e., whether the claim is supported, refuted, or cannot be verified from the provided evidence—depending on the dataset’s label set). We compare two approaches:

1. Prompting: The base LLM is not updated. Instead, we craft prompts that include the claim, supporting evidence, and a small set of selected training examples to guide the model toward producing the correct class label. 

2. Fine-tuning: We update the model parameters on the training split to specialize the LLM for the fact-checking labels. This includes training configuration choices such as epochs, batch size, and quantization settings, and we evaluate generalization on the validation split.

Both methods are evaluated on the same dataset partitions using standard classification metrics (Accuracy, Precision, Recall, F1 Macro, and F1 Weighted), and results are compared to understand the trade-offs between prompting and fine-tuning for fact-checking.

## Dataset Collection

In the ```data/``` folder, you can find a more detailed description of the dataset. You have access to the train and validation partitions. The test partition is reserved for hidden evaluation and is not available to you.

## Prompting the model

The prompting script is located at ```student/llm_prompting.py```. You must complete the empty classes, methods and functions.

**Example usage**

```
python student/llm_prompting.py \
  --host https://ollama.ux.uis.no \
  --model qwen3:0.6b \
  --train ./data/factchecking_train_data.json \
  --test ./data/factchecking_val_data.json \
  --doc_chars 1200 \
  --max_samples 100 \
  --seed 42
```

* ```--host``` → Specifies the Ollama server URL where the model is hosted.

* ```--model``` → Defines which language model to use for evaluation (e.g., qwen3:0.6b).

* ```--train``` → Path to the training dataset used for in-context learning example selection.

* ```--test``` → Path to the evaluation dataset (typically the validation set).

* ```--doc_chars``` → Maximum number of characters from the document/evidence to include in the prompt (used for truncation).

* ```--max_samples``` → Limits the number of evaluation samples. Use 0 to evaluate the entire dataset.

* ```--seed``` → Sets the random seed for reproducibility (e.g., sampling and example selection).

It is recommended that you create a ```.env``` file and store your ```OLLAMA_API_KEY```. To generate your API key:

* Log in to Open WebUI at https://openwebui.ux.uis.no using your UiS credentials.

* Navigate to Settings → Account.

* Generate a new API key.

* Copy and store your ```OLLAMA_API_KEY```.securely.

* Never commit your API key to version control (e.g., GitHub).

* Then you can save in ```.env``` file and call it using ```os``` library in Python.

Additionally, in order to successfully pass the prompting tests in GitHub Actions, you must store your ```OLLAMA_API_KEY``` as a repository secret.

To do this: 

* In your repository, go to Settings → Secrets and variables.

* Select Actions, then navigate to Repository secrets. 

* Click New repository secret.

* Set the Name to ```OLLAMA_API_KEY```.

* Paste your API key into the Secret field.

* Save the secret.

This allows GitHub Actions to securely access your API key during automated testing.

## Fine Tunning the model

The prompting script is located at ```student/llm_finetune.py```. You must complete the empty classes, methods and functions.

**Example Usage**

```
python student/llm_finetune.py \ 
    --num_epochs 5 \ 
    --batch_size 8 \
    --quantization 4bit
```

* ```--num_epochs``` → Number of full passes over the training dataset during fine-tuning. More epochs may improve learning but can also lead to overfitting.

* ```--batch_size``` → Number of training examples processed per step on each device. Larger batch sizes may speed up training but require more GPU memory.

* ```--quantization``` → Specifies the model quantization mode (e.g., 4bit

To successfully complete the assignment, you must correctly implement all required classes and methods, pass the provided implementation and the hidden unit tests, and achieve satisfactory performance on the test set. Your model will be evaluated using Accuracy, F1 (Weighted), F1 (Macro), Precision, and Recall, and all of these metrics will be considered during grading.

For more details, see the docstrings in each script.

## Important

Due to server and GPU usage limitations, you must pass the prompting and fine-tuning tests separately. Each time you make a commit related to prompting, the corresponding GitHub Actions workflow for prompting will be triggered. The same applies to fine-tuning: commits related to fine-tuning will trigger its dedicated workflow. Because GPU resources are limited, it is strongly recommended that you only push commits when a substantial portion of your implementation is completed and ready for evaluation. The prompting code does not require GPU resources, so you are encouraged to run and test it locally on your own machine before committing.
