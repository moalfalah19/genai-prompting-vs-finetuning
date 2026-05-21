#!/usr/bin/env python3
"The label tells the model what the correct answer is. During training, "
"the model compares its prediction with the label, computes the loss, and updates its weights. "
"In this way, the model learns to predict the correct label from the input."


from __future__ import annotations

import unsloth
import json
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers import TrainingArguments, BitsAndBytesConfig
from unsloth import FastLanguageModel
from trl import SFTTrainer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


@dataclass
class FineTuneConfig:
    # Training and validation file paths
    train_path: str = "data/factchecking_train_data.json"
    test_path: str = "data/factchecking_val_data.json"

    # Base model and output locations
    model_name: str = "unsloth/Qwen3-0.6B"
    output_dir: str = "./outputs"
    save_dir: str = "./finetuned_model"
    eval_results_path: str = "./evaluation_results_finetunning.json"

    # Main training settings
    train_fraction: float = ...
    num_epochs: int = ...
    batch_size: int = ...
    max_seq_length: int = ...
    lora_r: int = ...
    disable_eval: bool = False
    quantization: str = "4bit"

    # Generation / evaluation settings
    max_new_tokens: int = 10
    temperature: float = 0.1
    top_p: float = 0.9

    # Evaluation output settings
    eval_doc_chars: int = 500
    save_predictions_n: int = 10
    seed: int = 42

#It sends the input to the model, gets the output, reads the loss from the model, and returns it to the trainer.”
# We can rebuild it as long as we still return the correct loss. The important part is that the trainer must know how to compute the loss from the model outputs.
class SafeSFTTrainer(SFTTrainer):
    # Custom loss function used during supervised fine-tuning
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            labels=inputs.get("labels", None),
        )

        # Loss is computed by the model
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class FactCheckingDataLoader:
    # Loads JSON data and prepares training prompts
    def __init__(self, seed: int = 42):
        self.seed = seed

    def load_json(self, path: str) -> List[Dict[str, Any]]:
        # Check that the file exists
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        # Load JSON content
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def sample_fraction(self, data: List[Dict[str, Any]], fraction: float) -> List[Dict[str, Any]]:
        # Use only part of the training set if needed
        if not (0.0 < fraction <= 1.0):
            raise ValueError("--train_fraction must be in (0.0, 1.0].")

        if fraction >= 1.0:
            return data

        rng = random.Random(self.seed)
        sample_size = max(1, int(len(data) * fraction))
        return rng.sample(data, sample_size)

    @staticmethod

    # The label is the correct answer (SUPPORTS / REFUTES / MIXED)
    # It is included in the training text so the model learns what it should output
    # The model compares its prediction with this label to compute the loss
    def format_prompt(sample: Dict[str, Any]) -> str:
        # Convert one sample into training text
        claim = sample["claim"]
        label = sample["label"]
        doc = sample["doc"]

        return f"""Fact-check this claim based on the document.

Claim: {claim}
Document: {doc}

Label: {label}"""

    def to_hf_dataset(self, data: List[Dict[str, Any]]) -> Dataset:
        # Convert data into Hugging Face dataset
        return Dataset.from_dict({"text": [self.format_prompt(d) for d in data]})

#LoRA means Low-Rank Adaptation. It fine-tunes a model by adding small trainable matrices instead of changing all original weights.
#We use LoRA to make fine-tuning faster and more efficient.
# Without LoRA, we would have to train the entire model. This is very expensive, requires a lot of GPU memory, and is much slower.
class UnslothLoRATrainer:
    # Loads model, applies LoRA, builds trainer, trains, and saves model
    def __init__(self, cfg: FineTuneConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _build_quant_config(self) -> Tuple[Optional[BitsAndBytesConfig], bool]:
        # Build quantization settings
        if self.cfg.quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            return quant_config, True

        elif self.cfg.quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            return quant_config, False

        else:
            return None, False

    def load_model(self) -> None:
        # Create quantization config
        quant_config, load_in_4bit = self._build_quant_config()

        # Load pretrained model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cfg.model_name,
            max_seq_length=self.cfg.max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            quantization_config=quant_config,
        )

        # LoRA allows us to train only small matrices instead of the full model
        # Here we add LoRA adapters, and these are the parameters that get updated during training.
        model = FastLanguageModel.get_peft_model(
            model,

            # r is LoRA rank
            #this is controls the size of the small adapter matrices that we train. A higher rank means more learning capacity but also more computation and memory.
            #An adapter is a small part we add to the model to learn the task, instead of changing the whole model.
            r=self.cfg.lora_r,

            #target_modules are the parts of the transformer where LoRA is applied. Instead of training the whole model, we only modify these specific layers.
            # two main parts: 1- Attention part: These layers control how the model understands relationships between words.
            # 2- Feed-forward part: These layers help transform and refine the information after attention.
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],

            # Scaling factor for LoRA updates
            lora_alpha=16,

            # No dropout inside LoRA adapters
            lora_dropout=0,

            # Do not train bias terms
            bias="none",

            # Saves memory during training
            use_gradient_checkpointing="unsloth",

            # For reproducibility
            random_state=self.cfg.seed,
        )

        self.model = model
        self.tokenizer = tokenizer

        # Set padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use right-side padding
        if self.tokenizer.padding_side is None:
            self.tokenizer.padding_side = "right"

    def build_trainer(self, train_dataset: Dataset) -> None:
        # Model must be loaded first
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Training hyperparameters
        args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            num_train_epochs=self.cfg.num_epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            logging_steps=10,
            save_strategy="no",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            learning_rate=2e-4,
            fp16=True,
            bf16=False,
            seed=self.cfg.seed,
            report_to="none",
        )

        # Create trainer for supervised fine-tuning
        self.trainer = SafeSFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            args=args,
            max_seq_length=self.cfg.max_seq_length,
            dataset_text_field="text",
            packing=True,
        )

# The code does not update itself. It updates the model’s trainable parameters based on the loss.
#Backpropagation is the method used to compute gradients of the loss with respect to the model parameters.

    # During training the model repeatedly:
    # 1) It takes input and makes a prediction (forward pass)
    # 2) It compares the prediction with the correct label and computes the loss
    # 3) It calculates gradients using backpropagation
    # 4) It updates the weights using the optimizer
    def train(self) -> None:
        if self.trainer is None:
            raise RuntimeError("Trainer not built. Call build_trainer() first.")

        self.trainer.train()

    def save(self) -> None:
        # Save model and tokenizer
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        self.model.save_pretrained(self.cfg.save_dir)
        self.tokenizer.save_pretrained(self.cfg.save_dir)
        print(f"[INFO] Model saved to {self.cfg.save_dir}")


#class runs the model on test data, compares predictions with the true labels, and computes evaluation metrics like accuracy, F1 score, precision, and recall.
# inside this calss It sends each example to the model, gets the predicted label, compares it with the correct label, and calculates performance metrics.
#if we delete we cannot measure how well the model works, because we lose accuracy and other evaluation metrics.
class FactCheckingEvaluator:
    # Possible output labels
    LABELS = ["SUPPORTS", "REFUTES", "MIXED"]

    def __init__(self, cfg: FineTuneConfig):
        self.cfg = cfg

    @staticmethod
    def _normalize_label_text(text: str) -> str:
        # Clean spaces and lowercase text
        return " ".join(text.strip().split()).lower()

    def _extract_label(self, pred_text: str) -> str:
        # Convert generated text into one of the 3 labels
        t = self._normalize_label_text(pred_text)

        if "supports" in t or "support" in t:
            return "SUPPORTS"
        if "refutes" in t or "refute" in t:
            return "REFUTES"
        if "mixed" in t:
            return "MIXED"

        # Extra fallback rules
        if any(x in t for x in ["true", "supported", "yes"]):
            return "SUPPORTS"
        if any(x in t for x in ["false", "no"]):
            return "REFUTES"

        return "MIXED"

    def evaluate_split(
        self,
        model,
        tokenizer,
        data: List[Dict[str, Any]],
        split_name: str,
    ) -> Dict[str, Any]:
        # Evaluation mode disables training behavior like dropout
        model.eval()

        correct = 0
        total = 0
        predictions: List[Dict[str, Any]] = []
        true_labels: List[str] = []
        pred_labels: List[str] = []

        print(f"\n[INFO] Evaluating on {split_name} ({len(data)} samples)...")

        step = max(1, len(data) // 10)

        for i, sample in enumerate(data):
            if i % step == 0:
                print(f"  Progress: {i}/{len(data)}")

            claim = sample["claim"]
            gold = sample["label"]

            # Shorten document during evaluation
            doc = (sample["doc"] or "")[: self.cfg.eval_doc_chars]

            # Prompt ends with Label: so the model generates the answer
            eval_prompt = f"""Fact-check this claim based on the document.

Claim: {claim}
Document: {doc}

Label:"""

            # Tokenize prompt and move it to the same device as the model
            inputs = tokenizer(eval_prompt, return_tensors="pt").to(model.device)

            # No gradients needed during evaluation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                )

            # Decode model output
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Keep only the generated part after the prompt
            raw_tail = full_text[len(eval_prompt):].strip()

            # Convert generated text into label
            pred_label = self._extract_label(raw_tail)

            predictions.append(
                {
                    "claim": claim,
                    "actual_label": gold,
                    "predicted_label": pred_label,
                    "raw_prediction": self._normalize_label_text(raw_tail),
                }
            )

            true_labels.append(gold)
            pred_labels.append(pred_label)

            if pred_label == gold:
                correct += 1
            total += 1

        # Compute evaluation metrics
        accuracy = (correct / total * 100.0) if total else 0.0
        f1_weighted = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
        f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
        precision = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
        recall = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)

        print(f"\n{split_name} Results:")
        print(f"  Accuracy:    {accuracy:.2f}% ({correct}/{total})")
        print(f"  F1-Weighted: {f1_weighted:.4f}")
        print(f"  F1-Macro:    {f1_macro:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print("\n  Classification Report:")
        print(classification_report(true_labels, pred_labels, zero_division=0))

        return {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "precision": precision,
            "recall": recall,
            "predictions": predictions,
        }


class FineTunePipeline:
    # Runs the full fine-tuning and evaluation process
    def __init__(self, cfg: FineTuneConfig):
        self.cfg = cfg
        self.data = FactCheckingDataLoader(seed=cfg.seed)
        self.trainer = UnslothLoRATrainer(cfg)
        self.evaluator = FactCheckingEvaluator(cfg)

    def run(self) -> None:
        # Load and optionally sample training data
        train_data = self.data.load_json(self.cfg.train_path)
        train_data = self.data.sample_fraction(train_data, self.cfg.train_fraction)

        if self.cfg.train_fraction < 1.0:
            print(f"[INFO] Training on {len(train_data)} samples ({self.cfg.train_fraction*100:.1f}% of original)")

        # Convert training data to Hugging Face dataset
        train_ds = self.data.to_hf_dataset(train_data)

        # Load model and prepare trainer
        self.trainer.load_model()
        self.trainer.build_trainer(train_ds)

        # Run fine-tuning
        try:
            self.trainer.train()
        except Exception as e:
            print(f"[ERROR] Training error: {e}")
            import traceback
            traceback.print_exc()
            import sys
            sys.exit(1)

        # Save fine-tuned model
        self.trainer.save()

        # Skip evaluation if requested
        if self.cfg.disable_eval:
            print("[INFO] Skipping evaluation (--disable_eval)")
            return

        print("\n" + "=" * 80)
        print("EVALUATION")
        print("=" * 80)

        # Load test data and evaluate
        test_data = self.data.load_json(self.cfg.test_path)
        test_results = self.evaluator.evaluate_split(
            self.trainer.model,
            self.trainer.tokenizer,
            test_data,
            "Test",
        )

        # Save main metrics and a few predictions
        eval_results = {
            "test": {
                "accuracy": test_results["accuracy"],
                "f1_weighted": test_results["f1_weighted"],
                "f1_macro": test_results["f1_macro"],
                "precision": test_results["precision"],
                "recall": test_results["recall"],
                "predictions": test_results["predictions"][: self.cfg.save_predictions_n],
            },
        }

        with open(self.cfg.eval_results_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)

        print(f"\n[OK] Evaluation complete! Results saved to {self.cfg.eval_results_path}")


def parse_args() -> FineTuneConfig:
    # Read parameters from command line
    parser = argparse.ArgumentParser(description="Fine-tune LLM on factchecking data (class-based)")
    parser.add_argument("--train_path", type=str, default="data/factchecking_train_data.json")
    parser.add_argument("--test_path", type=str, default="data/factchecking_val_data.json")

    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--disable_eval", action="store_true")

    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"])

    parser.add_argument("--eval_doc_chars", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--save_predictions_n", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return FineTuneConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        train_fraction=args.train_fraction,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        disable_eval=args.disable_eval,
        quantization=args.quantization,
        eval_doc_chars=args.eval_doc_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        save_predictions_n=args.save_predictions_n,
        seed=args.seed,
    )


def main() -> None:
    # Create config and run pipeline
    cfg = parse_args()
    pipeline = FineTunePipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()