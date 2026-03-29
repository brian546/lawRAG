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

Edit `config.json` to set inference parameters, then run:

```bash
uv run scripts/inference.py
```

Or use a custom config:

```bash
uv run scripts/inference.py --config path/to/custom.json
```

Output:

- `artifacts/submission.csv`

### 📊 3) Optional train/val evaluation

Set `"evaluate": true` in config.json under the `inference` section, then run:

```bash
uv run scripts/inference.py
```

## 🧠 Fine-Tuning (Optional)

Configure `config.json` in the `fine_tune` section, then run:

```bash
uv run scripts/fine_tune.py
```

Common configurations:

```json
// config.json - dense only
"fine_tune": {
  "skip_reranker": true,
  ...
}

// config.json - reranker only
"fine_tune": {
  "skip_dense": true,
  ...
}

// config.json - QLoRA mode
"fine_tune": {
  "adapter_mode": "qlora",
  ...
}

// config.json - smoke test (no training)
"fine_tune": {
  "disable_lora": true,
  "skip_dense": true,
  "skip_reranker": true,
  ...
}
```

## 🔬 Sweep (Optional)

Configure `config.json` in the `sweep` section with `stage_k_values`, `out_k_values`, and MLflow settings, then run:

```bash
uv run scripts/sweep.py
```

## 🗂️ Configuration

All scripts load default settings from `config.json` at the project root. Any CLI flag explicitly passed overrides the corresponding JSON value.

The file is split into four top-level sections:

| Section     | Consumed by                                               |
| ----------- | --------------------------------------------------------- |
| `shared`    | all scripts — sets `data_dir`, `artifact_dir`, and `seed` |
| `inference` | `scripts/inference.py`                                    |
| `fine_tune` | `scripts/fine_tune.py`                                    |
| `sweep`     | `scripts/sweep.py`                                        |

Annotated defaults:

```json
{
  "shared": {
    "data_dir": "data", // input dataset directory
    "artifact_dir": "artifacts", // output root for models, corpora, submissions
    "seed": 42
  },
  "inference": {
    "submission_name": "submission.csv",
    "restrict_to_labeled": false,
    "enable_chunking": false,
    "chunk_chars": 3600,
    "overlap_chars": 400,
    "dense_model_name": "intfloat/multilingual-e5-base",
    "reranker_model_name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "local_dense_model_path": null, // set to override with a fine-tuned artifact
    "local_reranker_model_path": null, // set to override with a fine-tuned artifact
    "stage_k": 150, // dense retrieval candidate pool size
    "out_k": 20, // final top-k predictions
    "threshold": 0.5,
    "use_reranker": false,
    "reranker_top_n": 50,
    "reranker_batch_size": 32,
    "evaluate": false
  },
  "fine_tune": {
    "restrict_to_labeled": false,
    "enable_chunking": false,
    "chunk_chars": 3600,
    "overlap_chars": 400,
    "dense_model_name": "intfloat/multilingual-e5-base",
    "reranker_model_name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "gradient_accumulation_steps": 1,
    "eval_steps": 100,
    "save_steps": 100,
    "save_total_limit": 2,
    "negatives_per_query": 3,
    "adapter_mode": "lora", // "lora" or "qlora"
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": "all-linear",
    "dense_output_dir": "artifacts/models/dense_ft",
    "reranker_output_dir": "artifacts/models/reranker_ft",
    "disable_lora": false,
    "skip_dense": false,
    "skip_reranker": false
  },
  "sweep": {
    "restrict_to_labeled": true,
    "dense_model_name": "intfloat/multilingual-e5-base",
    "reranker_model_name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "use_reranker": false,
    "reranker_top_n": 50,
    "threshold": 0.5,
    "stage_k_values": [100, 150],
    "out_k_values": [8, 12],
    "mlflow_tracking_uri": "http://127.0.0.1:5000",
    "mlflow_experiment": "lawrag-dense-sweep"
  }
}
```

Pass `--config path/to/custom.json` to any script to use an alternative config file.

## 📝 Notes

- Inference can auto-wire local model artifacts from `artifacts/models/dense_ft` and `artifacts/models/reranker_ft`.
- Exact citation normalization and deterministic formatting are built into shared utilities to preserve Macro F1 behavior.
- Full LoRA/QLoRA training requires CUDA GPU.
