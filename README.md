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

## 🗂️ Configuration

All scripts load default settings from `config.json` at the project root. Any CLI flag explicitly passed overrides the corresponding JSON value.

The file is split into four top-level sections:

| Section | Consumed by |
|---------|-------------|
| `shared` | all scripts — sets `data_dir`, `artifact_dir`, and `seed` |
| `inference` | `scripts/inference.py` |
| `fine_tune` | `scripts/fine_tune.py` |
| `sweep` | `scripts/sweep.py` |

Annotated defaults:

```json
{
  "shared": {
    "data_dir": "data",          // input dataset directory
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
    "local_dense_model_path": null,    // set to override with a fine-tuned artifact
    "local_reranker_model_path": null, // set to override with a fine-tuned artifact
    "stage_k": 150,       // dense retrieval candidate pool size
    "out_k": 20,          // final top-k predictions
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
    "adapter_mode": "lora",          // "lora" or "qlora"
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

## ⚙️ Key CLI Options

All scripts read their default settings from `config.json` in the project root. CLI flags override the JSON values when provided.

### `scripts/inference.py`

- `--config`: Optional path to a JSON config file. Defaults to the repository-level `config.json`.
- `--data-dir`: Path to the input dataset directory containing train/val/test CSV files.
- `--artifact-dir`: Directory where generated corpus files, cached artifacts, and submissions are written.
- `--submission-name`: Output filename for the generated submission CSV inside the artifact directory.
- `--restrict-to-labeled`: Restrict retrieval candidates to citations seen in labeled training data.
- `--enable-chunking`: Split long legal documents into chunks before indexing them for retrieval.
- `--chunk-chars`: Maximum character length for each chunk when chunking is enabled.
- `--overlap-chars`: Number of overlapping characters between adjacent chunks to preserve context.
- `--dense-model-name`: Hugging Face model name used for dense retrieval embeddings.
- `--reranker-model-name`: Hugging Face cross-encoder model name used for reranking.
- `--local-dense-model-path`: Optional local path to a fine-tuned dense retriever artifact.
- `--local-reranker-model-path`: Optional local path to a fine-tuned reranker artifact.
- `--stage-k`: Number of initial candidates retrieved before optional reranking.
- `--out-k`: Maximum number of citations returned as final predictions.
- `--threshold`: Score cutoff used to filter low-confidence predictions.
- `--use-reranker`: Enable the reranking stage after dense retrieval.
- `--reranker-top-n`: Number of top dense candidates passed into the reranker.
- `--reranker-batch-size`: Batch size used while scoring candidate pairs with the reranker.
- `--evaluate`: Also compute train and validation metrics before generating the test submission.

### `scripts/fine_tune.py`

- `--config`: Optional path to a JSON config file. Defaults to the repository-level `config.json`.
- `--data-dir`: Path to the dataset directory used to build training and validation pairs.
- `--artifact-dir`: Base artifact directory used while building the training corpus.
- `--restrict-to-labeled`: Restrict corpus construction to labeled citations only.
- `--dense-model-name`: Base sentence-transformer model used for dense retriever fine-tuning.
- `--reranker-model-name`: Base cross-encoder model used for reranker fine-tuning.
- `--epochs`: Number of full passes through the fine-tuning dataset.
- `--batch-size`: Per-step batch size used during training.
- `--learning-rate`: Optimizer learning rate for LoRA or QLoRA fine-tuning.
- `--warmup-ratio`: Fraction of training steps reserved for learning-rate warmup.
- `--eval-steps`: Frequency, in training steps, for running intermediate evaluation.
- `--save-steps`: Frequency, in training steps, for writing checkpoints.
- `--save-total-limit`: Maximum number of checkpoints retained on disk.
- `--negatives-per-query`: Number of negative citation examples sampled for each query.
- `--adapter-mode {lora,qlora}`: Select whether adapters are trained with standard LoRA or QLoRA.
- `--disable-lora`: Disable all fine-tuning stages and only run preprocessing or smoke-test setup.
- `--dense-output-dir`: Output directory for the fine-tuned dense retriever.
- `--reranker-output-dir`: Output directory for the fine-tuned reranker.
- `--skip-dense`: Skip dense retriever training and only run the reranker stage.
- `--skip-reranker`: Skip reranker training and only run the dense retriever stage.

## 📝 Notes

- Inference can auto-wire local model artifacts from `artifacts/models/dense_ft` and `artifacts/models/reranker_ft`.
- Exact citation normalization and deterministic formatting are built into shared utilities to preserve Macro F1 behavior.
- Full LoRA/QLoRA training requires CUDA GPU.
