# LawRAG Production Scripts

## Goal

Build an offline, reproducible legal citation retrieval system for the Kaggle competition "LLM Agentic Legal Information Retrieval".

For each English legal query, the system predicts the most relevant Swiss legal citations in the required semicolon-separated format, optimized for citation-level Macro F1.

## Work Done

- Refactored notebook experiments into production-ready Python modules and CLI scripts.
- Implemented a unified corpus builder for laws and court considerations.
- Added EDA pipeline that informs retrieval design and data coverage decisions.
- Added fine-tuning pipelines for dense retriever and reranker (with optional LoRA).
- Implemented dense retrieval with FAISS and multilingual sentence embeddings.
- Implemented optional cross-encoder reranking for better precision.
- Added citation normalization, aggregation, and deterministic prediction formatting.
- Added end-to-end inference pipeline that generates Kaggle-ready `submission.csv`.

## Project Layout

- `lawrag/common.py`
  - Text normalization, citation parsing utilities, deterministic seeding.
- `lawrag/data.py`
  - Dataset path helpers, labeled citation loading, unified corpus construction, optional chunking.
- `lawrag/retrieval.py`
  - Dense retriever (FAISS + sentence-transformers), optional LoRA adapter-aware model loading, cross-encoder reranker.
- `lawrag/pipeline.py`
  - End-to-end citation retrieval pipeline and Macro-F1 metrics.
- `lawrag/finetune.py`
  - Fine-tuning pair construction plus dense and reranker training routines.
- `scripts/eda.py`
  - Production EDA workflow with CSV reports and saved figures.
- `scripts/fine_tune.py`
  - Fine-tunes bi-encoder and/or cross-encoder from CLI.
- `scripts/inference.py`
  - Runs inference (optional eval) and writes submission CSV.

## Context

Problem summary:

- Input: English legal queries.
- Retrieval target: Swiss legal citations (statutes and court decisions, often in German).
- Required prediction format: semicolon-separated citation strings per query.
- Evaluation: citation-level Macro F1 (per-query F1 averaged across queries).
- Execution constraint: offline reproducible notebook/runtime (no internet), bounded runtime (Kaggle code-competition constraints).

How this repository addresses that problem:

- Corpus unification: combines federal laws and court considerations into one searchable citation corpus.
- Deterministic preprocessing: text normalization and citation parsing reduce formatting noise that hurts exact-match evaluation.
- High-recall retrieval: dense FAISS retrieval over multilingual sentence embeddings captures semantic matches across English query and German citation context.
- Precision improvement: optional cross-encoder reranking improves top-candidate ordering before final citation output.
- Citation-level aggregation: merges chunk-level hits back to citation-level scores (`max` or `mean`) aligned with citation-set evaluation.
- Reproducible training/inference: CLI scripts support local/offline execution and produce deterministic artifacts (`submission.csv`, EDA reports, model outputs).

## EDA Findings and Why They Matter

The EDA is used here as a decision framework for retrieval design, not as an end in itself.

Core interpretation:

- The competition behaves like a long-tail retrieval problem. Many citations are rare, so frequency-driven shortcuts are fragile. This supports a semantic retriever (dense embeddings) as the main recall engine.
- Query difficulty is highly variable. Some questions are straightforward while others require broader retrieval context. This motivates a two-stage setup: high-recall candidate generation followed by precision-focused reranking.
- The number of relevant citations per query is not constant. Because scoring is per-query Macro F1, this creates a direct precision-recall tradeoff. The pipeline therefore keeps output depth controllable (`out_k`, reranker cutoffs, aggregation strategy).
- Train/validation citation overlap is limited, indicating non-trivial distribution shift. This means generalization matters more than memorizing frequent citations from training data.
- Not all gold citations are present in the provided retrieval corpus, which imposes an upper bound on achievable score for some samples. This reframes errors: some misses are corpus-limited, not purely model-limited.
- Citation matching is exact-string based in evaluation, so normalization and deterministic formatting are part of model quality, not just data hygiene.

What this means for this project:

- Dense multilingual retrieval is prioritized for robust recall.
- Cross-encoder reranking is optional but important for precision gains under compute limits.
- Citation-level aggregation and deterministic formatting are treated as first-class components because they directly affect Macro F1.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 1) EDA Script

Run:

```bash
python scripts/eda.py --data-dir data --output-dir artifacts/eda
```

Outputs include:

- `artifacts/eda/dataset_summary.csv`
- `artifacts/eda/eda_report.csv`
- `artifacts/eda/citation_frequency.csv`
- `artifacts/eda/top_citations.csv`
- `artifacts/eda/citations_missing_from_corpus.csv`
- PNG plots for query length, citation count distributions, and correlation scatter.

Useful options:

```bash
python scripts/eda.py --data-dir data --output-dir artifacts/eda --top-n-citations 30
```

Flag reference (`scripts/eda.py`):

- `--data-dir`:
  - Path to input CSV files (`train.csv`, `val.csv`, `test.csv`, `laws_de.csv`, `court_considerations.csv`).
  - Default: `data`
- `--output-dir`:
  - Directory where EDA CSV reports and PNG plots are written.
  - Default: `artifacts/eda`
- `--top-n-citations`:
  - Number of most frequent citations to store in `top_citations.csv`.
  - Default: `20`

## 2) Fine-Tuning Script

Default training (dense + reranker):

```bash
python scripts/fine_tune.py \
  --data-dir data \
  --restrict-to-labeled \
  --dense-output-dir artifacts/models/dense_ft \
  --reranker-output-dir artifacts/models/reranker_ft
```

Common variants:

- Dense only:

```bash
python scripts/fine_tune.py --data-dir data --restrict-to-labeled --skip-reranker
```

- Reranker only:

```bash
python scripts/fine_tune.py --data-dir data --restrict-to-labeled --skip-dense
```

- Disable LoRA:

```bash
python scripts/fine_tune.py --data-dir data --restrict-to-labeled --disable-lora
```

Key tuning flags:

- `--epochs`, `--batch-size`, `--learning-rate`, `--warmup-ratio`
- `--eval-steps`, `--save-steps`, `--save-total-limit`
- `--negatives-per-query`
- `--enable-chunking --chunk-chars 1200 --overlap-chars 200`

Flag reference (`scripts/fine_tune.py`):

- `--data-dir`:
  - Root directory containing training/validation/test and corpus CSV files.
  - Default: `data`
- `--restrict-to-labeled`:
  - If set, corpus is filtered to citations observed in train/val labels.
  - Default: disabled
- `--enable-chunking`:
  - If set, splits long corpus texts into overlapping chunks before training pair construction.
  - Default: disabled
- `--chunk-chars`:
  - Target chunk size in characters when chunking is enabled.
  - Default: `1200`
- `--overlap-chars`:
  - Character overlap between consecutive chunks.
  - Default: `200`
- `--dense-model`:
  - Dense bi-encoder base model name or local path.
  - Default: `intfloat/multilingual-e5-base`
- `--reranker-model`:
  - Cross-encoder reranker base model name or local path.
  - Default: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- `--seed`:
  - Global random seed for pair building and trainer reproducibility.
  - Default: `42`
- `--epochs`:
  - Number of training epochs for each enabled trainer.
  - Default: `3`
- `--batch-size`:
  - Per-device training batch size.
  - Default: `16`
- `--learning-rate`:
  - Optimizer learning rate.
  - Default: `2e-5`
- `--warmup-ratio`:
  - Fraction of total steps used for LR warmup.
  - Default: `0.1`
- `--eval-steps`:
  - Evaluation interval in training steps.
  - Default: `100`
- `--save-steps`:
  - Checkpoint save interval in training steps.
  - Default: `100`
- `--save-total-limit`:
  - Maximum number of checkpoints to keep.
  - Default: `2`
- `--disable-lora`:
  - If set, trains full model heads/backbone instead of applying LoRA adapters.
  - Default: disabled (LoRA enabled)
- `--lora-r`:
  - LoRA rank parameter.
  - Default: `32`
- `--lora-alpha`:
  - LoRA scaling alpha.
  - Default: `64`
- `--lora-dropout`:
  - LoRA dropout probability.
  - Default: `0.05`
- `--lora-bias`:
  - LoRA bias handling mode.
  - Default: `none`
- `--lora-target-modules`:
  - Space-separated module names where LoRA adapters are injected.
  - Default: `query key value dense`
- `--negatives-per-query`:
  - Number of random negatives sampled per query for cross-encoder training pair generation.
  - Default: `3`
- `--dense-output-dir`:
  - Output directory for the fine-tuned dense retriever.
  - Default: `artifacts/models/dense_ft`
- `--reranker-output-dir`:
  - Output directory for the fine-tuned reranker.
  - Default: `artifacts/models/reranker_ft`
- `--skip-dense`:
  - Skip dense model fine-tuning stage.
  - Default: disabled
- `--skip-reranker`:
  - Skip reranker fine-tuning stage.
  - Default: disabled

## 3) Inference Script

Generate submission:

```bash
python scripts/inference.py \
  --data-dir data \
  --artifact-dir artifacts \
  --submission-name submission.csv \
  --restrict-to-labeled \
  --use-reranker
```

Evaluate train/val before test submission:

```bash
python scripts/inference.py \
  --data-dir data \
  --artifact-dir artifacts \
  --restrict-to-labeled \
  --use-reranker \
  --evaluate
```

Use fine-tuned local models explicitly:

```bash
python scripts/inference.py \
  --data-dir data \
  --artifact-dir artifacts \
  --local-dense-model-path artifacts/models/dense_ft \
  --local-reranker-model-path artifacts/models/reranker_ft \
  --use-reranker
```

If local model paths are omitted, the script auto-detects:

- `artifacts/models/dense_ft`
- `artifacts/models/reranker_ft`

Flag reference (`scripts/inference.py`):

- `--data-dir`:
  - Root directory containing all input CSV files.
  - Default: `data`
- `--artifact-dir`:
  - Directory for generated outputs (submission and related artifacts).
  - Default: `artifacts`
- `--submission-name`:
  - Output submission filename inside `artifact-dir`.
  - Default: `submission.csv`
- `--seed`:
  - Seed used for deterministic pipeline behavior where applicable.
  - Default: `42`
- `--restrict-to-labeled`:
  - Restrict corpus to train/val labeled citations before retrieval indexing.
  - Default: disabled
- `--enable-chunking`:
  - Enable chunking of corpus texts prior to indexing.
  - Default: disabled
- `--chunk-chars`:
  - Chunk size in characters when chunking is enabled.
  - Default: `1200`
- `--overlap-chars`:
  - Chunk overlap in characters.
  - Default: `200`
- `--dense-model`:
  - Dense model name/path used when no local model override is provided.
  - Default: `intfloat/multilingual-e5-base`
- `--reranker-model`:
  - Reranker model name/path used when no local model override is provided.
  - Default: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- `--local-dense-model-path`:
  - Explicit local path to a fine-tuned dense model directory.
  - Default: `None` (auto-detect under `artifacts/models/dense_ft`)
- `--local-reranker-model-path`:
  - Explicit local path to a fine-tuned reranker model directory.
  - Default: `None` (auto-detect under `artifacts/models/reranker_ft`)
- `--stage-k`:
  - Number of dense retrieval candidates before citation aggregation/reranking.
  - Default: `150`
- `--out-k`:
  - Number of final predicted citations per query.
  - Default: `12`
- `--citation-score-agg`:
  - Aggregation strategy for chunk-level citation scores.
  - Choices: `max`, `mean`
  - Default: `max`
- `--use-reranker`:
  - Enable cross-encoder reranking stage.
  - Default: disabled
- `--reranker-top-n`:
  - Number of top aggregated candidates sent to reranker.
  - Default: `50`
- `--reranker-batch-size`:
  - Batch size used during reranker scoring.
  - Default: `32`
- `--evaluate`:
  - Compute and print train/val Macro-F1 before generating test submission.
  - Default: disabled

## Notebook-to-Script Mapping

- `kaggle_offline_submission.ipynb`:
  - pipeline + retrieval -> `lawrag/retrieval.py`, `lawrag/pipeline.py`
  - corpus preparation -> `lawrag/data.py`
  - fine-tuning -> `lawrag/finetune.py`, `scripts/fine_tune.py`
  - inference/submission -> `scripts/inference.py`

## Notes for Offline / Kaggle-style Runs

- Keep model references local using `--local-dense-model-path` and `--local-reranker-model-path`.
- For strict offline operation, ensure all models are present in local storage/datasets before execution.
- Runtime can be controlled via `--stage-k`, `--out-k`, and reranker options.
