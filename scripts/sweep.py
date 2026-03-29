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
    parser.add_argument("--config", default=None, help="Path to JSON config file.")
    return parser.parse_args()


def _parse_int_list(raw: str | list[int]) -> list[int]:
    if isinstance(raw, list):
        return [int(value) for value in raw]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    shared_config = get_config_section(load_json_config(args.config), "shared")
    sweep_config = get_config_section(load_json_config(args.config), "sweep")

    seed = int(shared_config.get("seed", SEED))
    set_global_seed(seed)

    if mlflow is None:
        raise RuntimeError("mlflow is not installed. Install it or skip sweep.")

    corpus_df, _ = build_and_save_corpus(
        config_data_dir=shared_config.get("data_dir", "data"),
        artifact_dir=shared_config.get("artifact_dir", "artifacts"),
        restrict_to_labeled=sweep_config.get("restrict_to_labeled", True),
        enable_chunking=sweep_config.get("enable_chunking", False),
        chunk_chars=sweep_config.get("chunk_chars", 3600),
        overlap_chars=sweep_config.get("overlap_chars", 400),
    )

    paths = DataPaths(root=Path(shared_config.get("data_dir", "data")))
    train_df = pd.read_csv(paths.train)
    val_df = pd.read_csv(paths.val)

    mlflow_tracking_uri = sweep_config.get("mlflow_tracking_uri")
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(sweep_config.get("mlflow_experiment", "lawrag-dense-sweep"))

    stage_k_values = _parse_int_list(sweep_config.get("stage_k_values", [100, 150]))
    out_k_values = _parse_int_list(sweep_config.get("out_k_values", [8, 12]))

    best_stage_k: int | None = None
    best_out_k: int | None = None
    best_train_f1: float | None = None
    best_val_f1: float | None = None
    for stage_k in stage_k_values:
        for out_k in out_k_values:
            cfg = PipelineConfig(
                mode=sweep_config.get("mode", "dense"),
                dense_model_name=sweep_config.get("dense_model_name", "intfloat/multilingual-e5-base"),
                stage_k=stage_k,
                out_k=out_k,
                use_reranker=sweep_config.get("use_reranker", False),
                reranker_model_name=sweep_config.get("reranker_model_name", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
                reranker_top_n=sweep_config.get("reranker_top_n", 50),
                threshold=sweep_config.get("threshold", 0.5),
            )

            pipeline = CitationRAGPipeline(
                corpus_df=corpus_df,
                config=cfg,
                local_dense_model_path=sweep_config.get("local_dense_model_path"),
                local_reranker_model_path=sweep_config.get("local_reranker_model_path"),
            )

            run_name = f"dense_sweep_{stage_k}_{out_k}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(
                    {
                        "mode": "dense",
                        "stage_k": stage_k,
                        "out_k": out_k,
                        "use_reranker": sweep_config.get("use_reranker", False),
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
