# Sentence Pair Classification

As an alternative to supervised approaches, this project contains three prompt-based classifiers for the detection of contradictions and inconsistencies between regulatory sentences. The accompanying paper can be found [here](https://scholarspace.manoa.hawaii.edu/server/api/core/bitstreams/5b99bd69-e789-4864-9cd8-4fe3d912b55a/content). The description of the collection, preprocessing, and annotation of the [data](https://github.com/schumanng/contradictions_inconsistencies) used is detailed in the paper [Detection of Contradictions and Inconsistencies in German Regulatory Documents](https://ieeexplore.ieee.org/abstract/document/10692679?casa_token=XZbIx1qqiCUAAAAA:T2CxqFoLAvT6902_9aU3EEdSAFfRsHQPDgryu_HORDXIRk8x7w4X8lozdjaxJc_wDlojc5Wlv7cd4w). 

The script 'run_ollama_classification.py' evaluates annotated sentence pairs against multiple models and multiple prompt variants.

---

# Project Purpose

Given two sentences, the models classify their relationship into predefined labels:

- Contradiction
- Inconsistent
- Neutral
- Not related

The framework supports:

- Multiple Ollama models
- Multiple prompt variants
- English and German datasets
- Automatic result export to CSV
- Logging of long-running experiments

---

# Project Structure

```text
project/
├── data/
│   ├── English/
│   │   └── annotated_regulatory_sentences_english.csv
│   └── German/
│       └── annotated_regulatory_sentences_german.csv
│
├── prompts/
│   ├── English/
│   │   ├── prompt_variant_1.txt
│   │   ├── prompt_variant_2.txt
│   │   └── prompt_variant_3.txt
│   │
│   └── German/
│       ├── prompt_variant_1.txt
│       ├── prompt_variant_2.txt
│       └── prompt_variant_3.txt
│
├── run_ollama_classification.py
└── README.md
```

> results/ and logs/ are created automatically during runtime.

---

# Requirements

Install Python packages:

```bash
pip install pandas ollama
```

Install and run Ollama:

https://ollama.com/download

Make sure the required models are available locally.

Example:

```bash
ollama pull llama3.3:70b-instruct-fp16
```

---

# Supported Languages

The script currently supports:

- English
- German

Language selection determines:

- which dataset is loaded
- which prompt folder is used
- where results are stored
- where logs are written

---

# How to Run

## English Experiment

```bash
python run_ollama_classification.py --language English
```

## German Experiment

```bash
python run_ollama_classification.py --language German
```

---

# Output Files

For each model-prompt combination, one CSV file is created.

Example:

```text
English__llama3.1_8b-instruct-fp16__prompt_variant_1__predictions.csv
```

Each CSV contains:

- sentence1
- sentence2
- gold_label
- prediction
- model
- prompt
- language
- is_correct

---

# Configuration

All major settings can be adjusted inside the CONFIG dictionary in run_ollama_classification.py.

Examples:

- models to test
- prompt files
- context window (num_ctx)
- GPU visibility
- CSV separator

---

# Notes

- Ollama must be running before execution.
- Long experiments may take several hours depending on dataset size and models.
- Intermediate results are saved regularly during runtime.
- The results are not deterministic.
- For research purposes only.

