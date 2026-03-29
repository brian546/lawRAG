#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from lawrag.common import (
    SEED,
    DataPaths,
    PipelineConfig,
    get_config_section,
    load_json_config,
    resolve_options,
    set_global_seed,
)
from lawrag.data import build_and_save_corpus
from lawrag.pipeline import CitationRAGPipeline, macro_f1


try:
    import mlflow
except ImportError:
    mlflow = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep stage_k/out_k and log to MLflow.")
    parser.add_argument("--config", default=None, help="Path to shared JSON config file.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--dense-model-name", default=None)
    parser.add_argument("--reranker-model-name", default=None)
    parser.add_argument("--local-dense-model-path", default=None)
    parser.add_argument("--local-reranker-model-path", default=None)
    parser.add_argument("--use-reranker", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reranker-top-n", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--stage-k-values", default=None)
    parser.add_argument("--out-k-values", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--mlflow-experiment", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _parse_int_list(raw: str | list[int]) -> list[int]:
    if isinstance(raw, list):
        return [int(value) for value in raw]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    file_config = get_config_section(load_json_config(args.config), "sweep")
    resolved = resolve_options(
        args,
        file_config,
        [
            "data_dir",
            "artifact_dir",
            "dense_model_name",
            "reranker_model_name",
            "local_dense_model_path",
            "local_reranker_model_path",
            "use_reranker",
            "reranker_top_n",
            "threshold",
            "stage_k_values",
            "out_k_values",
            "mlflow_tracking_uri",
            "mlflow_experiment",
            "seed",
            "restrict_to_labeled",
            "enable_chunking",
            "chunk_chars",
            "overlap_chars",
            "mode",
        ],
    )

    seed = int(resolved.get("seed", SEED))
    set_global_seed(seed)

    if mlflow is None:
        raise RuntimeError("mlflow is not installed. Install it or skip sweep.")

    corpus_df, _ = build_and_save_corpus(
        config_data_dir=resolved.get("data_dir", "data"),
        artifact_dir=resolved.get("artifact_dir", "artifacts"),
        restrict_to_labeled=resolved.get("restrict_to_labeled", True),
        enable_chunking=resolved.get("enable_chunking", False),
        chunk_chars=resolved.get("chunk_chars", 3600),
        overlap_chars=resolved.get("overlap_chars", 400),
    )

    paths = DataPaths(root=Path(resolved.get("data_dir", "data")))
    train_df = pd.read_csv(paths.train)
    val_df = pd.read_csv(paths.val)

    if resolved.get("mlflow_tracking_uri"):
        mlflow.set_tracking_uri(resolved["mlflow_tracking_uri"])
    mlflow.set_experiment(resolved.get("mlflow_experiment", "lawrag-dense-sweep"))

    stage_k_values = _parse_int_list(resolved.get("stage_k_values", [100, 150]))
    out_k_values = _parse_int_list(resolved.get("out_k_values", [8, 12]))

    best_stage_k: int | None = None
    best_out_k: int | None = None
    best_train_f1: float | None = None
    best_val_f1: float | None = None
    for stage_k in stage_k_values:
        for out_k in out_k_values:
            cfg = PipelineConfig(
                mode=resolved.get("mode", "dense"),
                dense_model_name=resolved.get("dense_model_name", "intfloat/multilingual-e5-base"),
                stage_k=stage_k,
                out_k=out_k,
                use_reranker=resolved.get("use_reranker", False),
                reranker_model_name=resolved.get("reranker_model_name", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
                reranker_top_n=resolved.get("reranker_top_n", 50),
                threshold=resolved.get("threshold", 0.5),
            )

            pipeline = CitationRAGPipeline(
                corpus_df=corpus_df,
                config=cfg,
                local_dense_model_path=resolved.get("local_dense_model_path"),
                local_reranker_model_path=resolved.get("local_reranker_model_path"),
            )

            run_name = f"dense_sweep_{stage_k}_{out_k}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(
                    {
                        "mode": "dense",
                        "stage_k": stage_k,
                        "out_k": out_k,
                        "use_reranker": resolved.get("use_reranker", False),
                    }
                )

                train_preds = [pipeline.predict(q) for q in train_df["query"].fillna("").tolist()]
                val_preds = [pipeline.predict(q) for q in val_df["query"].fillna("").tolist()]
                train_f1 = float(macro_f1(train_df["gold_citations"].fillna("").tolist(), train_preds))
                val_f1 = float(macro_f1(val_df["gold_citations"].fillna("").tolist(), val_preds))

                mlflow.log_metric("train_macro_f1", train_f1)
                mlflow.log_metric("val_macro_f1", val_f1)

                if best_val_f1 is None or val_f1 > best_val_f1:
                    best_stage_k = stage_k
                    best_out_k = out_k
                    best_train_f1 = train_f1
                    best_val_f1 = val_f1

    best = None
    if best_val_f1 is not None and best_stage_k is not None and best_out_k is not None and best_train_f1 is not None:
        best = {
            "stage_k": best_stage_k,
            "out_k": best_out_k,
            "train_macro_f1": best_train_f1,
            "val_macro_f1": best_val_f1,
        }

    print(f"Best sweep config: {best}")


if __name__ == "__main__":
    main()
