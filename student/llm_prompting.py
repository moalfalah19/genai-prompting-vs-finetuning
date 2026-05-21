#!/usr/bin/env python3

# Prompting vs Fine-tuning
#Prompting means we guide the model using instructions and examples in the input text, without changing the model itself. 
# The model uses its existing knowledge to generate answers.
# Fine-tuning means we train the model on a dataset and update its weights, so it learns the task better and improves its performance.

#System vs User prompt
#The system prompt defines the behavior and role of the model, meaning it tells the model what it is and how it should respond.
#The user prompt contains the actual task or input we want the model to solve, such as the claim and document in this case.

#Zero-shot vs Few-shot
#Zero-shot means we give the model only instructions without any examples, so it relies on its existing knowledge. 
#Few-shot means we provide a few examples in the prompt, which helps the model understand the pattern and perform the task more accurately.”
#often gives a higher score because the examples help the model understand the task better.”

from __future__ import annotations

import os
import json
import re
import random
import argparse
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from tqdm import tqdm

from ollama import Client, ResponseError


# --------------------------------------------------
# Environment Setup
# --------------------------------------------------

# Here we load environment variables from a .env file.
# The reason for this is that the program needs an API key to communicate with the Ollama server where the model runs.
# Instead of writing the API key directly inside the code (which would be insecure), we store it in a .env file and load it using load_dotenv().
# This is also required for GitHub Actions because the API key is stored as a repository secret during automated tests.
try:
    load_dotenv()
except ModuleNotFoundError:
    pass


# --------------------------------------------------
# Label Definitions
# --------------------------------------------------

# These are the three labels used for the classification task. The model must decide the relationship between a claim and a document.
# SUPPORTS -> the document confirms the claim
# REFUTES  -> the document contradicts the claim
# MIXED    -> the document does not clearly support or refute the claim
LABELS = ["SUPPORTS", "REFUTES", "MIXED"]

# We also convert the list of labels into a set. Checking membership in a set is faster than in a list.
LABEL_SET = set(LABELS)

# This variable will store the few-shot examples that we select from the training dataset.
# Few-shot examples are important because they show the model how the task works before it predicts the real example.
FEW_SHOT_EXAMPLES: List[Dict[str, str]] = []


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------

# This function loads a dataset from a JSON file.
# Each element in the dataset contains:
# - a claim
# - a document
# - a label
# We also check that the file actually contains a list, because the rest of the program expects the dataset to be formatted that way.
def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list.")

    return data


# This function cleans the text by removing unnecessary whitespace.
# Datasets sometimes contain extra spaces or line breaks.
# Cleaning the text makes the prompt easier for the model to read and avoids wasting tokens.
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# This function shortens long documents.
# Large language models cannot process unlimited text.
# If the document is too long:
# - the prompt may exceed the context limit
# - the model becomes slower
# - important information may be diluted
#
# Therefore we keep only the first N characters of the document.
def truncate(s: str, n: int) -> str:
    return (s or "")[:n]


# --------------------------------------------------
# Few-shot Example Selection
# --------------------------------------------------

# Few-shot prompting is a technique where we show the model a few solved examples before asking it to solve the real task.
#
# For example:
# CLAIM: ...
# DOCUMENT: ...
# LABEL: REFUTES

# By seeing these examples, the model learns the pattern of the task without updating its weights.
# This function selects those examples from the training dataset.
def select_few_shot_examples(
    train_data: List[Dict[str, Any]],
    seed: int = 42,
    k_per_label: int = 1,
    doc_chars: int = 350,
) -> List[Dict[str, str]]:

    # Random generator with a fixed seed so the selected examples are always the same.
    rng = random.Random(seed)

    # We group training examples by their label. This ensures we pick examples from all three classes.
    by_label: Dict[str, List[Dict[str, Any]]] = {label: [] for label in LABELS}

   #Loop through the training dataset to collect examples that we can later use as few-shot examples in the prompt.
    for ex in train_data:
        label = ex.get("label")

        # Check that the example has a valid label (SUPPORTS, REFUTES, or MIXED). This prevents incorrect or missing labels from being used.
        if label in LABEL_SET:

            # Extract the claim and document text and clean them by removing extra spaces or line breaks.
            claim = normalize_ws(ex.get("claim", ""))
            doc = normalize_ws(ex.get("doc", ""))
            # Make sure both claim and document exist before adding the example. This avoids empty or incomplete training samples.
            if claim and doc:
                by_label[label].append(
                    {
                        "claim": claim,
                        "doc": truncate(doc, doc_chars),
                        "label": label,
                    }
                )

    selected: List[Dict[str, str]] = []

    # We select a fixed number of examples for each label. In the final implementation we used two examples per label, which means the prompt contains:
    # 2 SUPPORTS examples, 2 REFUTES examples, 2 MIXED examples
    # This keeps the prompt balanced and helps the model learn the classification pattern.
    for label in LABELS:
        candidates = by_label[label]

        # We sort examples by document length. Shorter documents are preferred because they keep the prompt shorter.
        candidates.sort(key=lambda x: len(x["doc"]))

        selected.extend(candidates[:k_per_label])

    return selected


# --------------------------------------------------
# Ollama Model Wrapper
# --------------------------------------------------

# This class manages the connection to the language model hosted on the Ollama server.
# Instead of calling the API everywhere in the code, we use this class to keep the interaction with the model organized.
class OllamaLLM:

    def __init__(self, host: str, model: str) -> None:

        # The API key is read from the environment variables.
        api_key = os.getenv("OLLAMA_API_KEY")

        if not api_key:
            raise RuntimeError("OLLAMA_API_KEY is not set")

        self.host = host
        self.model = model

        # Create a client that communicates with the Ollama server.
        self._client = Client(
            host=self.host,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    # This function sends a prompt to the model and returns the generated response.
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:

        # These parameters control the model generation.
        # temperature = 0 makes the output deterministic, which is useful for classification tasks.
        options = kwargs.get(
            "options",
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "num_predict": 80,
            },
        )

        try:

            response = self._client.chat(
                model=self.model,
                messages=messages,
                options=options,
                think=False,
            )

        except ResponseError as e:

            # If the model is not available on the server, we automatically download it.
            if getattr(e, "status_code", None) == 404:
                self._client.pull(self.model)

                response = self._client.chat(
                    model=self.model,
                    messages=messages,
                    options=options,
                )
            else:
                raise

        # Extract the generated text from the response.
        if hasattr(response, "message"):
            msg = response.message

            if hasattr(msg, "content") and msg.content:
                return msg.content.strip()

        if isinstance(response, dict):
            if "message" in response:
                return response["message"].get("content", "").strip()

        return ""


# --------------------------------------------------
# Prompt Construction
# --------------------------------------------------

# This function builds the full prompt that will be sent to the model. The prompt contains three main parts:
# 1. System instruction 2. Few-shot examples 3. The actual claim and document to classify
def make_prompt(
    claim: str,
    doc: str,
    doc_chars: int = 1200,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, str]]:

    #We clean the text to remove extra spaces and make the input consistent. 
    #We also truncate the document to keep it within the model’s context limit and make the model faster.
    claim = normalize_ws(claim)
    doc = truncate(normalize_ws(doc), doc_chars)


    #The system prompt defines the role of the model and gives high-level instructions. It tells the model what task to perform and how to respond.
    system_prompt = (
        "You are a fact-checking classifier.\n"
        "Classify the relationship between a CLAIM and DOCUMENT.\n\n"
        "output EXACTLY ONE label: SUPPORTS or REFUTES or MIXED.\n"
        "Respond with only the label."
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    # Add the few-shot examples.
    # Each example is written as a dialogue:
    # user -> claim and document
    # assistant -> correct label
    # This format works well for chat-based models.

    #Few-shot examples show the model how the task works by giving examples of input and correct output. 
    #We use few-shot examples to help the model understand the task better by showing it examples. This usually improves accuracy and gives better results.
    for ex in FEW_SHOT_EXAMPLES:

        messages.append({
            "role": "user",
            "content": f"CLAIM: {ex['claim']}\nDOCUMENT: {ex['doc']}\nLABEL:"
        })

        messages.append({
            "role": "assistant",
            "content": ex["label"]
        })

    # Finally we add the real example that we want the model to classify.
    messages.append({
        "role": "user",
        "content": f"CLAIM: {claim}\nDOCUMENT: {doc}\nLABEL:"
    })

    return messages


# This regular expression searches for valid labels inside the model output.
#This extracts the correct label from the model output. If the model generates extra text, we still find the label.
LABEL_REGEX = re.compile(r"\b(SUPPORTS|REFUTES|MIXED)\b", re.IGNORECASE)


# This function extracts the predicted label from the model output.
# If the model produces extra text, the regex finds the label inside it.
# If no label is found, we default to MIXED to avoid breaking the pipeline.
def parse_label(raw: str) -> str:

    if not raw:
        return "MIXED"

    m = LABEL_REGEX.search(raw)

    if m:
        return m.group(1).upper()

    return "MIXED"