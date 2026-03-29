#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from lawrag.common import DataPaths, PipelineConfig, set_global_seed
from lawrag.data import build_and_save_corpus
from lawrag.pipeline import CitationRAGPipeline, macro_f1


try:
    import mlflow
except ImportError:
    mlflow = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep stage_k/out_k and log to MLflow.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--dense-model-name", default="intfloat/multilingual-e5-base")
    parser.add_argument("--reranker-model-name", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--local-dense-model-path", default=None)
    parser.add_argument("--local-reranker-model-path", default=None)
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--reranker-top-n", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--stage-k-values", default="100,150")
    parser.add_argument("--out-k-values", default="8,12")
    parser.add_argument("--mlflow-tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--mlflow-experiment", default="lawrag-dense-sweep")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    if mlflow is None:
        raise RuntimeError("mlflow is not installed. Install it or skip sweep.")

    corpus_df, _ = build_and_save_corpus(
        config_data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        restrict_to_labeled=True,
        enable_chunking=False,
        chunk_chars=3600,
        overlap_chars=400,
    )

    paths = DataPaths(root=Path(args.data_dir))
    train_df = pd.read_csv(paths.train)
    val_df = pd.read_csv(paths.val)

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    stage_k_values = _parse_int_list(args.stage_k_values)
    out_k_values = _parse_int_list(args.out_k_values)

    best: dict[str, float | int] | None = None
    for stage_k in stage_k_values:
        for out_k in out_k_values:
            cfg = PipelineConfig(
                mode="dense",
                dense_model_name=args.dense_model_name,
                stage_k=stage_k,
                out_k=out_k,
                use_reranker=args.use_reranker,
                reranker_model_name=args.reranker_model_name,
                reranker_top_n=args.reranker_top_n,
                threshold=args.threshold,
            )

            pipeline = CitationRAGPipeline(
                corpus_df=corpus_df,
                config=cfg,
                local_dense_model_path=args.local_dense_model_path,
                local_reranker_model_path=args.local_reranker_model_path,
            )

            run_name = f"dense_sweep_{stage_k}_{out_k}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(
                    {
                        "mode": "dense",
                        "stage_k": stage_k,
                        "out_k": out_k,
                        "use_reranker": args.use_reranker,
                    }
                )

                train_preds = [pipeline.predict(q) for q in train_df["query"].fillna("").tolist()]
                val_preds = [pipeline.predict(q) for q in val_df["query"].fillna("").tolist()]
                train_f1 = float(macro_f1(train_df["gold_citations"].fillna("").tolist(), train_preds))
                val_f1 = float(macro_f1(val_df["gold_citations"].fillna("").tolist(), val_preds))

                mlflow.log_metric("train_macro_f1", train_f1)
                mlflow.log_metric("val_macro_f1", val_f1)

                if best is None or val_f1 > float(best["val_macro_f1"]):
                    best = {
                        "stage_k": stage_k,
                        "out_k": out_k,
                        "train_macro_f1": train_f1,
                        "val_macro_f1": val_f1,
                    }

    print(f"Best sweep config: {best}")


if __name__ == "__main__":
    main()
