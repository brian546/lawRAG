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
    resolve_options,
    set_global_seed,
)
from lawrag.data import build_and_save_corpus
from lawrag.pipeline import CitationRAGPipeline, macro_f1, macro_precision, macro_recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LawRAG inference/evaluation/submission generation.")
    parser.add_argument("--config", default=None, help="Path to shared JSON config file.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--submission-name", default=None)

    parser.add_argument("--restrict-to-labeled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-chunking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--chunk-chars", type=int, default=None)
    parser.add_argument("--overlap-chars", type=int, default=None)

    parser.add_argument("--dense-model-name", default=None)
    parser.add_argument("--reranker-model-name", default=None)
    parser.add_argument("--local-dense-model-path", default=None)
    parser.add_argument("--local-reranker-model-path", default=None)

    parser.add_argument("--stage-k", type=int, default=None)
    parser.add_argument("--out-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)

    parser.add_argument("--use-reranker", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reranker-top-n", type=int, default=None)
    parser.add_argument("--reranker-batch-size", type=int, default=None)

    parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=None, help="Evaluate on train/val.")
    parser.add_argument("--seed", type=int, default=None)
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
    file_config = get_config_section(load_json_config(args.config), "inference")
    resolved = resolve_options(
        args,
        file_config,
        [
            "data_dir",
            "artifact_dir",
            "submission_name",
            "restrict_to_labeled",
            "enable_chunking",
            "chunk_chars",
            "overlap_chars",
            "dense_model_name",
            "reranker_model_name",
            "local_dense_model_path",
            "local_reranker_model_path",
            "stage_k",
            "out_k",
            "threshold",
            "use_reranker",
            "reranker_top_n",
            "reranker_batch_size",
            "evaluate",
            "seed",
            "mode",
        ],
    )

    base_cfg = Config()
    seed = int(resolved.get("seed", SEED))
    set_global_seed(seed)

    cfg = Config(
        data_dir=resolved.get("data_dir", base_cfg.data_dir),
        artifact_dir=resolved.get("artifact_dir", base_cfg.artifact_dir),
        submission_name=resolved.get("submission_name", base_cfg.submission_name),
        restrict_to_labeled_citations=resolved.get("restrict_to_labeled", base_cfg.restrict_to_labeled_citations),
        enable_chunking=resolved.get("enable_chunking", base_cfg.enable_chunking),
        chunk_chars=resolved.get("chunk_chars", base_cfg.chunk_chars),
        overlap_chars=resolved.get("overlap_chars", base_cfg.overlap_chars),
        mode=resolved.get("mode", base_cfg.mode),
        dense_model_name=resolved.get("dense_model_name", base_cfg.dense_model_name),
        reranker_model_name=resolved.get("reranker_model_name", base_cfg.reranker_model_name),
        local_dense_model_path=resolved.get("local_dense_model_path", base_cfg.local_dense_model_path),
        local_reranker_model_path=resolved.get("local_reranker_model_path", base_cfg.local_reranker_model_path),
        stage_k=resolved.get("stage_k", base_cfg.stage_k),
        out_k=resolved.get("out_k", base_cfg.out_k),
        threshold=resolved.get("threshold", base_cfg.threshold),
        use_reranker=resolved.get("use_reranker", base_cfg.use_reranker),
        reranker_top_n=resolved.get("reranker_top_n", base_cfg.reranker_top_n),
        reranker_batch_size=resolved.get("reranker_batch_size", base_cfg.reranker_batch_size),
    )

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

    if bool(resolved.get("evaluate", False)):
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
