# ⚖️ LawRAG

Modular, production-style legal citation retrieval pipeline for the Kaggle competition **LLM Agentic Legal Information Retrieval**.

The system predicts Swiss legal citations for English legal queries and outputs Kaggle-ready submissions evaluated with citation-level Macro F1.

## ✨ What Changed

`kaggle_offline_submission.ipynb` has been modularized into reusable Python modules and CLI scripts:

- Reusable package in `lawrag/`
- Script entrypoints in `scripts/`
- End-to-end inference, evaluation, fine-tuning, and sweep workflows runnable from terminal

## 🧱 Package Layout

```text
lawRAG/
├── lawrag/
│   ├── __init__.py
│   ├── common.py            # config dataclasses, text normalization, citation parsing
│   ├── data.py              # labeled citation set loading, unified corpus build, chunking
│   ├── retrieval.py         # model loaders, FAISS retriever, cross-encoder reranker
│   ├── pipeline.py          # LangGraph retrieval pipeline + Macro P/R/F1 metrics
│   └── finetune.py          # fine-tuning pair generation + dense/reranker training routines
├── scripts/
│   ├── inference.py         # corpus build, pipeline run, evaluation, submission generation
│   ├── fine_tune.py         # dense/reranker LoRA or QLoRA fine-tuning
│   └── sweep.py             # stage_k/out_k sweep with MLflow tracking
├── data/
├── artifacts/
├── kaggle_offline_submission.ipynb
├── eda.ipynb                # exploratory data analysis
├── pyproject.toml
└── README.md
```

## 🚀 Quick Start

### 🛠️ 1) Environment setup

```bash
# Install uv once (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and populate virtual environment from pyproject.toml
uv sync
```


### 📤 2) Generate submission (inference)

```bash
uv run scripts/inference.py \
  --data-dir data \
  --artifact-dir artifacts \
  --submission-name submission.csv \
  --restrict-to-labeled \
  --use-reranker
```

Output:

- `artifacts/submission.csv`

### 📊 3) Optional train/val evaluation

```bash
uv run scripts/inference.py \
  --data-dir data \
  --artifact-dir artifacts \
  --submission-name submission_eval.csv \
  --restrict-to-labeled \
  --use-reranker \
  --evaluate
```

## 🧠 Fine-Tuning (Optional)

Run dense + reranker fine-tuning:

```bash
uv run scripts/fine_tune.py \
  --data-dir data \
  --artifact-dir artifacts \
  --restrict-to-labeled \
  --dense-output-dir artifacts/models/dense_ft \
  --reranker-output-dir artifacts/models/reranker_ft
```

Common variants:

```bash
# Dense only
uv run scripts/fine_tune.py --data-dir data --restrict-to-labeled --skip-reranker

# Reranker only
uv run scripts/fine_tune.py --data-dir data --restrict-to-labeled --skip-dense

# QLoRA mode
uv run scripts/fine_tune.py --data-dir data --restrict-to-labeled --adapter-mode qlora

# Smoke test without training
uv run scripts/fine_tune.py --data-dir data --restrict-to-labeled --disable-lora --skip-dense --skip-reranker
```

## 🔬 Sweep (Optional)

Run stage_k/out_k sweep with MLflow:

```bash
uv run scripts/sweep.py \
  --data-dir data \
  --artifact-dir artifacts \
  --use-reranker \
  --stage-k-values 100,150 \
  --out-k-values 8,12 \
  --mlflow-tracking-uri http://127.0.0.1:5000 \
  --mlflow-experiment lawrag-dense-sweep
```

## ⚙️ Key CLI Options

### `scripts/inference.py`

- `--data-dir`, `--artifact-dir`, `--submission-name`
- `--restrict-to-labeled`
- `--enable-chunking`, `--chunk-chars`, `--overlap-chars`
- `--dense-model-name`, `--reranker-model-name`
- `--local-dense-model-path`, `--local-reranker-model-path`
- `--stage-k`, `--out-k`, `--threshold`
- `--use-reranker`, `--reranker-top-n`, `--reranker-batch-size`
- `--evaluate`

### `scripts/fine_tune.py`

- `--epochs`, `--batch-size`, `--learning-rate`, `--warmup-ratio`
- `--eval-steps`, `--save-steps`, `--save-total-limit`
- `--negatives-per-query`
- `--adapter-mode {lora,qlora}`
- `--skip-dense`, `--skip-reranker`, `--disable-lora`
- `--dense-output-dir`, `--reranker-output-dir`

## ✅ Validation Performed

The modularized code was validated in this workspace with:

- `uv sync --refresh`
- `uv run python -m py_compile lawrag/*.py scripts/*.py`
- `uv run scripts/inference.py ... --submission-name submission_smoke.csv`
- `python scripts/fine_tune.py ... --disable-lora --skip-dense --skip-reranker`
- `python scripts/sweep.py --help`

Generated smoke output:

- `artifacts/submission_smoke.csv`

## 📝 Notes

- Inference can auto-wire local model artifacts from `artifacts/models/dense_ft` and `artifacts/models/reranker_ft`.
- Exact citation normalization and deterministic formatting are built into shared utilities to preserve Macro F1 behavior.
- Full LoRA/QLoRA training requires CUDA GPU.
