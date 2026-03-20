#!/usr/bin/env python3
"""Inference and optional evaluation script for LawRAG pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from lawrag.common import set_seed
from lawrag.data import DataPaths, build_unified_corpus, chunk_corpus, load_labeled_citation_set
from lawrag.pipeline import CitationRAGPipeline, PipelineConfig, macro_f1

LOGGER = logging.getLogger("inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LawRAG inference and produce submission.csv")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--submission-name", type=str, default="submission.csv")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--restrict-to-labeled", action="store_true")
    parser.add_argument("--enable-chunking", action="store_true")
    parser.add_argument("--chunk-chars", type=int, default=1200)
    parser.add_argument("--overlap-chars", type=int, default=200)

    parser.add_argument("--dense-model", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )
    parser.add_argument("--local-dense-model-path", type=str, default=None)
    parser.add_argument("--local-reranker-model-path", type=str, default=None)

    parser.add_argument("--stage-k", type=int, default=150)
    parser.add_argument("--out-k", type=int, default=12)
    parser.add_argument("--citation-score-agg", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--reranker-top-n", type=int, default=50)
    parser.add_argument("--reranker-batch-size", type=int, default=32)

    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def maybe_resolve_local_models(args: argparse.Namespace) -> tuple[str | None, str | None]:
    dense_path = args.local_dense_model_path
    reranker_path = args.local_reranker_model_path

    default_dense = args.artifact_dir / "models" / "dense_ft"
    default_reranker = args.artifact_dir / "models" / "reranker_ft"
    if dense_path is None and default_dense.exists():
        dense_path = str(default_dense)
    if reranker_path is None and default_reranker.exists():
        reranker_path = str(default_reranker)
    return dense_path, reranker_path


def evaluate_split(pipeline: CitationRAGPipeline, df: pd.DataFrame, name: str) -> float:
    preds = [pipeline.predict(query) for query in df["query"].fillna("").tolist()]
    score = macro_f1(df["gold_citations"].fillna("").tolist(), preds)
    LOGGER.info("%s macro_f1=%.6f", name, score)
    return score


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    set_seed(args.seed)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = DataPaths(root=args.data_dir)

    allowed = load_labeled_citation_set(paths) if args.restrict_to_labeled else None
    corpus_df = build_unified_corpus(paths, allowed_citations=allowed)
    if args.enable_chunking:
        corpus_df = chunk_corpus(
            corpus_df,
            chunk_chars=args.chunk_chars,
            overlap_chars=args.overlap_chars,
        )

    dense_local, reranker_local = maybe_resolve_local_models(args)
    pipeline_cfg = PipelineConfig(
        mode="dense",
        dense_model_name=args.dense_model,
        citation_score_agg=args.citation_score_agg,
        stage_k=args.stage_k,
        out_k=args.out_k,
        use_reranker=args.use_reranker,
        reranker_model_name=args.reranker_model,
        reranker_top_n=args.reranker_top_n,
        reranker_batch_size=args.reranker_batch_size,
    )

    pipeline = CitationRAGPipeline(
        corpus_df=corpus_df,
        config=pipeline_cfg,
        local_dense_model_path=dense_local,
        local_reranker_model_path=reranker_local,
    )

    LOGGER.info("Pipeline initialized. stage_k=%d out_k=%d", args.stage_k, args.out_k)
    LOGGER.info("Dense model ref: %s", dense_local or args.dense_model)
    LOGGER.info("Reranker model ref: %s", reranker_local or args.reranker_model)

    if args.evaluate:
        train_df = pd.read_csv(paths.train)
        val_df = pd.read_csv(paths.val)
        evaluate_split(pipeline, train_df, "train")
        evaluate_split(pipeline, val_df, "val")

    test_df = pd.read_csv(paths.test)
    preds = [pipeline.predict(query) for query in test_df["query"].fillna("").tolist()]

    submission_df = pd.DataFrame(
        {
            "query_id": test_df["query_id"],
            "predicted_citations": preds,
        }
    )
    submission_path = args.artifact_dir / args.submission_name
    submission_df.to_csv(submission_path, index=False)
    LOGGER.info("Saved submission to %s", submission_path)


if __name__ == "__main__":
    main()
