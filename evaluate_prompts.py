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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

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
INFER_BATCH_SIZE = 4

PROMPTS = {
    "Author_Boolean": "Czy autor podanej wypowiedzi kłamał czy mówił prawdę? Jeśli mówił prawdę odpowiedz 1. Jeśli kłamał, 0. \nWypowiedź: {text}",
    "Author_Belief": "Czy autor podanej wypowiedzi wierzył, że mówi prawdę (wierzył w to co mówi)? 1=Tak, 0=Nie.\nWypowiedź: {text}",
    "Author_Mislead": "Czy autor podanej wypowiedzi miał intencję wprowadzenia w błąd? Odpowiedz 1 (Tak) lub 0 (Nie).\nWypowiedź: {text}",
    "Boolean_Single": "Czy podana wypowiedź jest prawdziwa? Odpowiedz 1 (Tak) lub 0 (Nie).\nWypowiedź: {text}",
    "Boolean_Majority": "Czy większośc zdań tej wypowiedzi jest prawdziwa? Jeśli przeważają prawdziwe zdania, zwróć 1. W przeciwnym razie 0.\nWypowiedź: {text}"
}


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
def apply_chatml_prompt(text: str, prompt_template: str, tokenizer) -> str:
    if len(text) > 8000:
        text = text[:8000]

    user_content = prompt_template.format(text=text)
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_hf_dataset(records: List[Dict], tokenizer, prompt_template: str):
    ds = Dataset.from_pandas(pd.DataFrame(records))

    def tokenize_fn(batch):
        prompts = [apply_chatml_prompt(t, prompt_template, tokenizer) for t in batch["text"]]
        return tokenizer(prompts, truncation=True, max_length=MAX_INPUT_TOKENS)

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


class StatAnalyzer:
    @staticmethod
    def compute_metrics(y_true, y_pred):
        return {
            "Acc": accuracy_score(y_true, y_pred),
            "Prec": precision_score(y_true, y_pred, zero_division=0),
            "Rec": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0)
        }

    @staticmethod
    def bootstrap_interval(y_true, y_pred, metric_func=f1_score, n_boot=1000):
        rng = np.random.RandomState(42)
        indices = np.arange(len(y_true))
        stats = []
        for _ in range(n_boot):
            sample = rng.choice(indices, size=len(indices), replace=True)
            stats.append(metric_func(np.array(y_true)[sample], np.array(y_pred)[sample], zero_division=0))
        return np.percentile(stats, 2.5), np.percentile(stats, 97.5)

    @staticmethod
    def compare_prompts(results_df: pd.DataFrame, ground_truth_col="y_true"):
        prompt_cols = [c for c in results_df.columns if c != ground_truth_col]

        print("\n" + "="*50)
        print("STATISTICAL SIGNIFICANCE ANALYSIS")
        print("="*50)

        correctness_data = {}
        for p in prompt_cols:
            correctness_data[p] = (results_df[p] == results_df[ground_truth_col]).astype(int)

        correctness_df = pd.DataFrame(correctness_data)

        q_result = cochrans_q(correctness_df)
        try:
            q_stat = q_result.statistic
            p_val_global = q_result.pvalue
        except AttributeError:
            q_stat = q_result[0]
            p_val_global = q_result[1]

        print(f"Global Cochran's Q Test: p-value = {p_val_global:.5f}")

        n_prompts = len(prompt_cols)
        matrix = pd.DataFrame(index=prompt_cols, columns=prompt_cols)

        print("\nComputing Pairwise McNemar's Tests...")
        for i in range(n_prompts):
            for j in range(n_prompts):
                p1 = prompt_cols[i]
                p2 = prompt_cols[j]

                if i == j:
                    matrix.loc[p1, p2] = "-"
                    continue

                y_true = results_df[ground_truth_col]
                y_pred1 = results_df[p1]
                y_pred2 = results_df[p2]

                c1 = (y_pred1 == y_true)
                c2 = (y_pred2 == y_true)

                a = ((c1) & (c2)).sum()
                b = ((c1) & (~c2)).sum()
                c = ((~c1) & (c2)).sum()
                d = ((~c1) & (~c2)).sum()

                table = [[a, b], [c, d]]
                result = mcnemar(table, exact=True)

                star = "*" if result.pvalue < 0.05 else ""
                matrix.loc[p1, p2] = f"{result.pvalue:.4f}{star}"

        return matrix


# ==============================================================================
# 3. MODEL BUILDER (INFERENCE-ONLY)
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
    return model


def predict_labels_in_batches(texts: List[str], prompt_template: str, tokenizer, model, batch_size: int) -> List[int]:
    device = next(model.parameters()).device
    preds: List[int] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        prompts = [apply_chatml_prompt(t, prompt_template, tokenizer) for t in batch_texts]
        enc = tokenizer(
            prompts,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            batch_preds = torch.argmax(outputs.logits, dim=-1).tolist()
        preds.extend(int(p) for p in batch_preds)
    return preds


def run_prompt_eval(input_pkl: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    raw = load_dataset(input_pkl)
    if not raw:
        return

    data = [{"label": int(y), "text": t[:MAX_INPUT_CHARS]} for (y, t) in raw]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(tokenizer)
    model.eval()

    prompt_summaries = []
    results_df = pd.DataFrame({"y_true": [x["label"] for x in data]})

    for prompt_name, prompt_template in PROMPTS.items():
        print(f"Prompt {prompt_name} - starting inference")

        texts = [x["text"] for x in data]
        preds = predict_labels_in_batches(texts, prompt_template, tokenizer, model, INFER_BATCH_SIZE)
        results_df[prompt_name] = preds

        y_true = results_df["y_true"].tolist()
        y_pred = results_df[prompt_name].tolist()
        metrics = StatAnalyzer.compute_metrics(y_true, y_pred)
        f1_lo, f1_hi = StatAnalyzer.bootstrap_interval(y_true, y_pred)

        prompt_summaries.append({
            "prompt": prompt_name,
            "Acc": metrics["Acc"],
            "Prec": metrics["Prec"],
            "Rec": metrics["Rec"],
            "F1": metrics["F1"],
            "F1_CI_low": f1_lo,
            "F1_CI_high": f1_hi,
        })

        print(f"Prompt {prompt_name} - inference complete")
        clean_memory()

    prompt_summary_df = pd.DataFrame(prompt_summaries).set_index("prompt")

    # Save outputs
    results_df.to_csv(os.path.join(output_dir, "prompt_predictions.csv"), index=False)
    prompt_summary_df.to_csv(os.path.join(output_dir, "prompt_summary.csv"))

    with open(os.path.join(output_dir, "prompt_summary.tex"), "w", encoding="utf-8") as f:
        f.write(prompt_summary_df.to_latex(float_format="%.4f", caption="Prompt-wise metrics with F1 CI"))

    # Pairwise significance matrix
    matrix = StatAnalyzer.compare_prompts(results_df)
    with open(os.path.join(output_dir, "pairwise_confidence.tex"), "w", encoding="utf-8") as f:
        f.write(matrix.to_latex(caption="Pairwise McNemar Test P-Values"))

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
            "infer_batch_size": INFER_BATCH_SIZE,
            "prompts": list(PROMPTS.keys()),
        }, f, indent=2)

    print("All prompts completed.")
    print(f"Saved: {os.path.join(output_dir, 'prompt_predictions.csv')}")
    print(f"Saved: {os.path.join(output_dir, 'prompt_summary.csv')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prompt evaluation with a non-finetuned model")
    parser.add_argument("--input", default="p.pkl", help="Path to input pickle")
    parser.add_argument("--output", default="outputs", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prompt_eval(args.input, args.output)
