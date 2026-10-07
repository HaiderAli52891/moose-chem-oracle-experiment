# MOOSE-Chem Oracle Experiment

Ablation script that measures the contribution of retrieval, composition,
and ranking to hypothesis quality on the MOOSE-Chem benchmark.

## What it does

Runs four conditions per paper:
- base                : normal retrieval -> normal composition -> LLM ranker
- oracle_retrieval    : gold inspirations -> normal composition -> LLM ranker
- oracle_composition  : normal retrieval -> stronger composition -> LLM ranker
- oracle_ranking      : normal retrieval -> normal composition -> oracle ranker

Then reports each stage's percentage contribution to final hypothesis quality.

## Requirements

- Python 3.10+
- Ollama running locally at http://localhost:11434
- Model: qwen-8k (or any Ollama chat model — change MODEL in the script)

Install Python dependencies:
pip install openpyxl

## Dataset

Uses the MOOSE-Chem benchmark dataset:
https://github.com/QQQ-cmd/MOOSE-Chem

Place `chem_research_2024.xlsx` in a local `Data/` folder and update
`DATA_FILE` in the script if needed. The dataset is not included in this
repository to respect the original authors' distribution terms.

## How to run

python oracle_experiment.py

Set `LIMIT = 3` for a quick pilot, `LIMIT = None` for the full benchmark.

## Status

Tested end-to-end on a 3-paper pilot with qwen-8k via Ollama on a local
GPU laptop.