import os
import gc
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import inspect
from typing import List, Dict

from datasets import Dataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
try:
    from transformers import EarlyStoppingCallback
except Exception:
    EarlyStoppingCallback = None
try:
    from transformers import TrainerCallback
except Exception:
    TrainerCallback = None

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# ==============================================================================
# CONFIGURATION (defaults)
# ==============================================================================
MODEL_ID = "speakleash/Bielik-11B-v2.3-Instruct"
MAX_INPUT_TOKENS = 1024
MAX_INPUT_CHARS = 2000
N_SPLITS = 10
DEV_SPLIT = 0.10
SEED = 42

# Fixed hyperparameters
LORA_R = 8
LORA_ALPHA = 32
LEARNING_RATE = 2e-4

BATCH_SIZE = 2
GRAD_ACCUM = 8
NUM_EPOCHS = 3
EARLY_STOPPING_PATIENCE = 2


# ==============================================================================
# 1. DATASET LOADER
# ==============================================================================
def load_dataset(file_path: str) -> List:
    """
    Expected Pickle format: List of tuples (label, text).
    Label must be integer 0 or 1.
    """
    print(f"Loading dataset from: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f" >> Loaded {len(data)} samples.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}.")
        return []


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ==============================================================================
# 2. TOKENIZATION / DATASET HELPERS
# ==============================================================================
def build_hf_dataset(records: List[Dict], tokenizer):
    ds = Dataset.from_pandas(pd.DataFrame(records))

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch")
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


class SimpleEarlyStoppingCallback(TrainerCallback if TrainerCallback is not None else object):
    def __init__(self, patience=2, metric="eval_f1", greater_is_better=True):
        self.patience = patience
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.best = None
        self.num_bad_epochs = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.metric not in metrics:
            return control

        value = metrics[self.metric]
        if self.best is None:
            self.best = value
            self.num_bad_epochs = 0
            return control

        improved = value > self.best if self.greater_is_better else value < self.best
        if improved:
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                control.should_training_stop = True
        return control


def build_training_args(output_dir: str):
    common = {
        "output_dir": output_dir,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "fp16": torch.cuda.is_available(),
    }

    sig = inspect.signature(TrainingArguments.__init__)

    if "eval_strategy" in sig.parameters:
        common["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig.parameters:
        common["evaluation_strategy"] = "epoch"

    if "save_strategy" in sig.parameters:
        common["save_strategy"] = "epoch"

    if "logging_strategy" in sig.parameters:
        common["logging_strategy"] = "epoch"

    if "report_to" in sig.parameters:
        common["report_to"] = "none"

    if "load_best_model_at_end" in sig.parameters:
        common["load_best_model_at_end"] = True
    if "metric_for_best_model" in sig.parameters:
        common["metric_for_best_model"] = "f1"
    if "greater_is_better" in sig.parameters:
        common["greater_is_better"] = True
    if "save_total_limit" in sig.parameters:
        common["save_total_limit"] = 1

    return TrainingArguments(**common)


def build_trainer(model, args, train_ds, eval_ds, tokenizer):
    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": DataCollatorWithPadding(tokenizer),
        "compute_metrics": compute_metrics,
    }

    sig = inspect.signature(Trainer.__init__)
    if "tokenizer" in sig.parameters:
        kwargs["tokenizer"] = tokenizer

    callbacks = []
    if EarlyStoppingCallback is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE))
    elif TrainerCallback is not None:
        callbacks.append(SimpleEarlyStoppingCallback(patience=EARLY_STOPPING_PATIENCE))

    if callbacks and "callbacks" in sig.parameters:
        kwargs["callbacks"] = callbacks

    return Trainer(**kwargs)


# ==============================================================================
# 3. MODEL BUILDER WITH QLoRA
# ==============================================================================
def build_model(tokenizer):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(model, lora_config)
    return model


# ==============================================================================
# 4. CROSS-VALIDATION WITH DEV SET EARLY STOPPING
# ==============================================================================
def run_supervised_cv(input_pkl: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    raw = load_dataset(input_pkl)
    if not raw:
        return

    data = [{"label": int(y), "text": t[:MAX_INPUT_CHARS]} for (y, t) in raw]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_metrics = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(data), 1):
        print(f"Fold {fold_idx}/{N_SPLITS} - starting")

        train_val = [data[i] for i in train_val_idx]
        test = [data[i] for i in test_idx]

        train, dev = train_test_split(
            train_val,
            test_size=DEV_SPLIT,
            random_state=SEED,
            stratify=[x["label"] for x in train_val],
        )

        model = build_model(tokenizer)
        train_ds = build_hf_dataset(train, tokenizer)
        dev_ds = build_hf_dataset(dev, tokenizer)

        args = build_training_args(output_dir=os.path.join(output_dir, f"fold_{fold_idx}"))
        trainer = build_trainer(model, args, train_ds, dev_ds, tokenizer)

        trainer.train()

        if hasattr(trainer, "_load_best_model"):
            try:
                trainer._load_best_model()
            except Exception:
                pass

        test_ds = build_hf_dataset(test, tokenizer)
        test_metrics = trainer.evaluate(eval_dataset=test_ds)

        fold_row = {
            "fold": fold_idx,
            "accuracy": test_metrics.get("eval_accuracy", 0.0),
            "precision": test_metrics.get("eval_precision", 0.0),
            "recall": test_metrics.get("eval_recall", 0.0),
            "f1": test_metrics.get("eval_f1", 0.0),
        }
        fold_metrics.append(fold_row)

        print(f"Fold {fold_idx} done: acc={fold_row['accuracy']:.4f} f1={fold_row['f1']:.4f} prec={fold_row['precision']:.4f} rec={fold_row['recall']:.4f}")

        del trainer, model
        clean_memory()

    df = pd.DataFrame(fold_metrics)
    mean = df.drop(columns=["fold"]).mean()
    stderr = df.drop(columns=["fold"]).std(ddof=1) / np.sqrt(len(df))

    summary = pd.DataFrame({
        "mean": mean,
        "stderr": stderr,
    })

    # Save outputs
    df.to_csv(os.path.join(output_dir, "cv_per_fold_metrics.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "cv_summary.csv"))

    with open(os.path.join(output_dir, "cv_summary.tex"), "w", encoding="utf-8") as f:
        f.write(summary.to_latex(float_format="%.4f", caption="10-fold CV metrics with standard error"))

    with open(os.path.join(output_dir, "cv_per_fold_metrics.tex"), "w", encoding="utf-8") as f:
        f.write(df.to_latex(float_format="%.4f", caption="Per-fold metrics"))

    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_id": MODEL_ID,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_input_chars": MAX_INPUT_CHARS,
            "n_splits": N_SPLITS,
            "dev_split": DEV_SPLIT,
            "seed": SEED,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "num_epochs": NUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        }, f, indent=2)

    print("All folds completed.")
    print(f"Saved: {os.path.join(output_dir, 'cv_per_fold_metrics.csv')}")
    print(f"Saved: {os.path.join(output_dir, 'cv_summary.csv')}")


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA 10-fold CV with early stopping")
    parser.add_argument("--input", default="p.pkl", help="Path to input pickle")
    parser.add_argument("--output", default="outputs", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_supervised_cv(args.input, args.output)
