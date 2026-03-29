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
    Config,
    DataPaths,
    PipelineConfig,
    get_config_section,
    load_json_config,
    set_global_seed,
)
from lawrag.data import build_and_save_corpus
from lawrag.pipeline import CitationRAGPipeline, macro_f1, macro_precision, macro_recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LawRAG inference/evaluation/submission generation.")
    parser.add_argument("--config", default=None, help="Path to JSON config file.")
    return parser.parse_args()


def evaluate_split(pipeline: CitationRAGPipeline, df: pd.DataFrame, name: str) -> dict[str, float]:
    preds = [pipeline.predict(q) for q in df["query"].fillna("").tolist()]
    gold = df["gold_citations"].fillna("").tolist()

    metrics = {
        "macro_precision": float(macro_precision(gold, preds)),
        "macro_recall": float(macro_recall(gold, preds)),
        "macro_f1": float(macro_f1(gold, preds)),
    }
    print(f"{name} metrics: {metrics}")
    return metrics


def auto_wire_local_models(cfg: Config) -> None:
    artifact_dir = Path(cfg.artifact_dir)
    dense_artifact_path = artifact_dir / "models" / "dense_ft"
    reranker_artifact_path = artifact_dir / "models" / "reranker_ft"

    dense_name_lc = str(cfg.dense_model_name).lower()
    looks_like_e5 = "e5" in dense_name_lc

    if cfg.local_dense_model_path is None and dense_artifact_path.exists() and looks_like_e5:
        cfg.local_dense_model_path = str(dense_artifact_path)
    if cfg.local_reranker_model_path is None and reranker_artifact_path.exists():
        cfg.local_reranker_model_path = str(reranker_artifact_path)


def main() -> None:
    args = parse_args()
    shared_config = get_config_section(load_json_config(args.config), "shared")
    inference_config = get_config_section(load_json_config(args.config), "inference")

    seed = int(shared_config.get("seed", SEED))
    set_global_seed(seed)

    cfg = Config(
        data_dir=shared_config.get("data_dir", "data"),
        artifact_dir=shared_config.get("artifact_dir", "artifacts"),
        submission_name=inference_config.get("submission_name", "submission.csv"),
        restrict_to_labeled_citations=inference_config.get("restrict_to_labeled", False),
        enable_chunking=inference_config.get("enable_chunking", False),
        chunk_chars=inference_config.get("chunk_chars", 3600),
        overlap_chars=inference_config.get("overlap_chars", 400),
        mode=inference_config.get("mode", "dense"),
        dense_model_name=inference_config.get("dense_model_name", "intfloat/multilingual-e5-base"),
        reranker_model_name=inference_config.get("reranker_model_name", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
        local_dense_model_path=inference_config.get("local_dense_model_path"),
        local_reranker_model_path=inference_config.get("local_reranker_model_path"),
        stage_k=inference_config.get("stage_k", 150),
        out_k=inference_config.get("out_k", 20),
        threshold=inference_config.get("threshold", 0.5),
        use_reranker=inference_config.get("use_reranker", False),
        reranker_top_n=inference_config.get("reranker_top_n", 50),
        reranker_batch_size=inference_config.get("reranker_batch_size", 32),
    )

    evaluate = inference_config.get("evaluate", False)

    corpus_df, corpus_path = build_and_save_corpus(
        config_data_dir=cfg.data_dir,
        artifact_dir=cfg.artifact_dir,
        restrict_to_labeled=cfg.restrict_to_labeled_citations,
        enable_chunking=cfg.enable_chunking,
        chunk_chars=cfg.chunk_chars,
        overlap_chars=cfg.overlap_chars,
    )
    print(f"Saved corpus: {corpus_path}")
    print(f"Corpus rows: {len(corpus_df):,}")

    auto_wire_local_models(cfg)

    pipeline_cfg = PipelineConfig(
        mode=cfg.mode,
        dense_model_name=cfg.dense_model_name,
        stage_k=cfg.stage_k,
        out_k=cfg.out_k,
        use_reranker=cfg.use_reranker,
        reranker_model_name=cfg.reranker_model_name,
        reranker_top_n=cfg.reranker_top_n,
        reranker_batch_size=cfg.reranker_batch_size,
        threshold=cfg.threshold,
    )

    pipeline = CitationRAGPipeline(
        corpus_df=corpus_df,
        config=pipeline_cfg,
        local_dense_model_path=cfg.local_dense_model_path,
        local_reranker_model_path=cfg.local_reranker_model_path,
    )

    paths = DataPaths(root=Path(cfg.data_dir))

    if bool(evaluate):
        train_df = pd.read_csv(paths.train)
        val_df = pd.read_csv(paths.val)
        evaluate_split(pipeline, train_df, "train")
        evaluate_split(pipeline, val_df, "val")

    test_df = pd.read_csv(paths.test)
    preds = [pipeline.predict(q) for q in test_df["query"].fillna("").tolist()]
    submission_df = pd.DataFrame({"query_id": test_df["query_id"], "predicted_citations": preds})

    out_dir = Path(cfg.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_path = out_dir / cfg.submission_name
    submission_df.to_csv(submission_path, index=False)
    print(f"Saved submission to: {submission_path}")


if __name__ == "__main__":
    main()
