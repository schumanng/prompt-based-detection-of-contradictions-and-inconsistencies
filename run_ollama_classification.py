"""
Before running this script, make sure Ollama is running and the configured models
are available locally.

Usage:
    python run_ollama_classification.py --language English
    python run_ollama_classification.py --language German
"""

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import ollama
import pandas as pd


# -------------
# Configuration

PROJECT_ROOT = Path(__file__).resolve().parent

CONFIG = {
    "data_dir": PROJECT_ROOT / "data",
    "prompt_root_dir": PROJECT_ROOT / "prompts",
    "output_root_dir": PROJECT_ROOT / "results",
    "log_root_dir": PROJECT_ROOT / "logs",

    "input_csv_files": {
        "English": "annotated_regulatory_sentences_english.csv",
        "German": "annotated_regulatory_sentences.csv",
    },

    # The script expects three prompt files and applies each prompt iteratively.
    "prompt_files": [
        "prompt_variant_1.txt",
        "prompt_variant_2.txt",
        "prompt_variant_3.txt",
    ],

    "csv_separator": ";",
    "ollama_num_ctx": 10000,

    # Set this to a comma-separated GPU list if needed, e.g. "0,1,2,3".
    # Leave as None if GPU visibility should be controlled outside the script.
    "cuda_visible_devices": "0,1,2,3,4,5,6,7",

    "models": [
        "llama3.1:8b-instruct-fp16",
        "llama3.3:70b-instruct-fp16",
        "llama3.1:405b-instruct-q8_0",
        "qwen2.5:72b-instruct-fp16",
        "deepseek-r1:70b-llama-distill-fp16",
    ],
}


# -----------------
# Command-line input

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Ollama-based sentence-pair classification experiments."
    )
    parser.add_argument(
        "--language",
        choices=["English", "German"],
        required=True,
        help="Language-specific data and prompt folder to use.",
    )
    return parser.parse_args()


# -----------------
# Utility functions

def write_log(message: str, language: str) -> None:
    """Append a timestamped message to the language-specific log file."""
    log_file = Path(CONFIG["log_root_dir"]) / language / "classification_run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")
    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] {message}\n")


def sanitize_filename(value: str) -> str:
    """Create a filesystem-safe identifier from a model, prompt, or language name."""
    value = value.replace(":", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")


def load_prompt(prompt_path: Path) -> str:
    """Read one prompt file from disk."""
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}. "
            "Create the file or adjust CONFIG['prompt_files']."
        )

    return prompt_path.read_text(encoding="utf-8").strip()


def build_user_content(sentence1: str, sentence2: str) -> str:
    """Format a sentence pair for the user message sent to Ollama."""
    return f"Sentence 1: {sentence1}\nSentence 2: {sentence2}"


def normalize_prediction(prediction: str) -> str:
    """Normalize model output for comparison against gold labels."""
    return str(prediction).strip().lower()


# ----------------
# Data preparation

def preprocess_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant columns, remove identical and duplicate sentence pairs,
    shuffle the dataset, and print a compact label distribution.
    """
    required_columns = ["sentence1", "sentence2", "gold_label"]
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {missing_columns}")

    data = data[required_columns]
    print("Dataset length (initial):", len(data))

    identical_mask = data["sentence1"].eq(data["sentence2"])
    data = data.loc[~identical_mask].copy()
    print("Dataset length (after removing identical sentence pairs):", len(data))

    data = data.drop_duplicates(subset=["sentence1", "sentence2"]).copy()
    print("Dataset length (after removing duplicate sentence pairs):", len(data))

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nPreview:")
    print(data.head(20))

    print("\nLabel distribution:")
    print(data["gold_label"].value_counts(dropna=False).reset_index(name="count"))

    return data


def load_dataset(language: str) -> pd.DataFrame:
    """Load and preprocess the language-specific annotated sentence-pair dataset."""
    input_csv = (
        Path(CONFIG["data_dir"])
        / language
        / CONFIG["input_csv_files"][language]
    )

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}. "
            "Adjust CONFIG['input_csv_files'] or your data folder structure."
        )

    data = pd.read_csv(input_csv, sep=CONFIG["csv_separator"])
    return preprocess_dataset(data)


# --------------
# Classification

def classify_sentence_pair(
    model: str,
    system_prompt: str,
    sentence1: str,
    sentence2: str,
) -> str:
    """Send one sentence pair to Ollama and return the raw model prediction."""
    response = ollama.chat(
        model=model,
        options={"num_ctx": CONFIG["ollama_num_ctx"]},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_content(sentence1, sentence2)},
        ],
    )
    return response["message"]["content"]


def run_predictions_for_model_and_prompt(
    data: pd.DataFrame,
    model: str,
    prompt_name: str,
    system_prompt: str,
    language: str,
) -> pd.DataFrame:
    """Run one model with one prompt over all sentence pairs."""
    results = []

    output_dir = Path(CONFIG["output_root_dir"]) / language
    output_dir.mkdir(parents=True, exist_ok=True)

    language_id = sanitize_filename(language)
    model_id = sanitize_filename(model)
    prompt_id = sanitize_filename(Path(prompt_name).stem)

    output_path = output_dir / f"{language_id}__{model_id}__{prompt_id}__predictions.csv"

    print("\n----------")
    print(f"Starting predictions | language={language} | model={model} | prompt={prompt_name}")
    write_log(
        f"Starting predictions | language={language} | model={model} | prompt={prompt_name}",
        language=language,
    )

    for idx, row in data.iterrows():
        sentence1 = row["sentence1"]
        sentence2 = row["sentence2"]
        gold_label = row["gold_label"]

        print(f"Sentence pair index: {idx}")
        print(sentence1, "|", sentence2, "|", gold_label)

        try:
            prediction = classify_sentence_pair(
                model=model,
                system_prompt=system_prompt,
                sentence1=sentence1,
                sentence2=sentence2,
            )
        except Exception as exc:
            prediction = f"ERROR: {exc}"
            write_log(
                f"Prediction failed | language={language} | model={model} | "
                f"prompt={prompt_name} | idx={idx} | error={exc}",
                language=language,
            )

        normalized_prediction = normalize_prediction(prediction)
        is_correct = normalized_prediction == str(gold_label).strip().lower()

        print("--->", prediction)
        print("correct prediction!" if is_correct else "wrong prediction!")
        print("-----------------")

        results.append(
            {
                "sentence1": sentence1,
                "sentence2": sentence2,
                "gold_label": gold_label,
                "prediction": normalized_prediction,
                "model": model,
                "prompt": prompt_id,
                "language": language,
                "is_correct": is_correct,
            }
        )

        # Periodically persist intermediate results to avoid losing progress
        # during long-running experiments.
        if (idx + 1) % 200 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

    write_log(f"Finished predictions | output={output_path}", language=language)
    return result_df


def run_all_experiments(data: pd.DataFrame, language: str) -> None:
    """Run all configured model-prompt combinations for one language."""
    prompt_dir = Path(CONFIG["prompt_root_dir"]) / language

    if not prompt_dir.exists():
        raise FileNotFoundError(
            f"Prompt directory not found: {prompt_dir}. "
            "Expected structure: prompts/English or prompts/German."
        )

    for prompt_file in CONFIG["prompt_files"]:
        prompt_path = prompt_dir / prompt_file
        system_prompt = load_prompt(prompt_path)

        for model in CONFIG["models"]:
            run_predictions_for_model_and_prompt(
                data=data,
                model=model,
                prompt_name=prompt_file,
                system_prompt=system_prompt,
                language=language,
            )


# ----------------
# Main entry point

def main() -> None:
    """Execute the complete classification experiment."""
    args = parse_args()
    language = args.language

    if CONFIG["cuda_visible_devices"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]

    write_log(f"Script started | language={language}", language=language)

    data = load_dataset(language)
    run_all_experiments(data, language)

    write_log(f"Script finished | language={language}", language=language)


if __name__ == "__main__":
    main()
