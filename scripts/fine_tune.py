#!/usr/bin/env python3
"""Fine-tuning script for dense retriever and reranker."""

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
from lawrag.finetune import (
    FinetuneConfig,
    build_finetune_pairs,
    run_dense_finetuning,
    run_reranker_finetuning,
)

LOGGER = logging.getLogger("fine_tune")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LawRAG bi-encoder and cross-encoder.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
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
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)

    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-bias", type=str, default="none")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default=["query", "key", "value", "dense"],
    )
    parser.add_argument("--negatives-per-query", type=int, default=3)

    parser.add_argument("--dense-output-dir", type=Path, default=Path("artifacts/models/dense_ft"))
    parser.add_argument(
        "--reranker-output-dir",
        type=Path,
        default=Path("artifacts/models/reranker_ft"),
    )
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    set_seed(args.seed)

    paths = DataPaths(root=args.data_dir)

    allowed = load_labeled_citation_set(paths) if args.restrict_to_labeled else None
    corpus_df = build_unified_corpus(paths, allowed_citations=allowed)
    if args.enable_chunking:
        corpus_df = chunk_corpus(
            corpus_df,
            chunk_chars=args.chunk_chars,
            overlap_chars=args.overlap_chars,
        )

    train_df = pd.read_csv(paths.train)
    val_df = pd.read_csv(paths.val)

    train_bi, train_ce = build_finetune_pairs(
        train_df,
        corpus_df,
        negatives_per_query=args.negatives_per_query,
        seed=args.seed,
    )
    val_bi, val_ce = build_finetune_pairs(
        val_df,
        corpus_df,
        negatives_per_query=args.negatives_per_query,
        seed=args.seed + 1,
    )

    cfg = FinetuneConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        use_lora=not args.disable_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        lora_target_modules=args.lora_target_modules,
        negatives_per_query=args.negatives_per_query,
        dense_output_dir=str(args.dense_output_dir),
        reranker_output_dir=str(args.reranker_output_dir),
        run_dense_finetune=not args.skip_dense,
        run_reranker_finetune=not args.skip_reranker,
    )

    if cfg.run_dense_finetune:
        LOGGER.info("Fine-tuning dense retriever with %d train pairs", len(train_bi))
        dense_path = run_dense_finetuning(
            cfg,
            dense_model_ref=args.dense_model,
            train_pairs=train_bi,
            val_pairs=val_bi,
            seed=args.seed,
        )
        LOGGER.info("Saved dense model to %s", dense_path)
    else:
        LOGGER.info("Skipping dense fine-tuning")

    if cfg.run_reranker_finetune:
        LOGGER.info("Fine-tuning reranker with %d train pairs", len(train_ce))
        reranker_path = run_reranker_finetuning(
            cfg,
            reranker_model_ref=args.reranker_model,
            train_pairs=train_ce,
            val_pairs=val_ce,
            seed=args.seed,
        )
        LOGGER.info("Saved reranker model to %s", reranker_path)
    else:
        LOGGER.info("Skipping reranker fine-tuning")


if __name__ == "__main__":
    main()
