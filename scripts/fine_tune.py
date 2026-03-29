#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from lawrag.common import Config, DataPaths, FinetuneConfig, SEED, set_global_seed
from lawrag.data import build_and_save_corpus
from lawrag.finetune import build_finetune_pairs, finetune_dense, finetune_reranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LawRAG dense retriever and reranker.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--restrict-to-labeled", action="store_true")

    parser.add_argument("--dense-model-name", default="intfloat/multilingual-e5-base")
    parser.add_argument("--reranker-model-name", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--negatives-per-query", type=int, default=3)

    parser.add_argument("--adapter-mode", choices=["lora", "qlora"], default="lora")
    parser.add_argument("--disable-lora", action="store_true", help="Disable all fine-tuning stages.")

    parser.add_argument("--dense-output-dir", default="artifacts/models/dense_ft")
    parser.add_argument("--reranker-output-dir", default="artifacts/models/reranker_ft")

    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    cfg = Config(
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        restrict_to_labeled_citations=args.restrict_to_labeled,
        dense_model_name=args.dense_model_name,
        reranker_model_name=args.reranker_model_name,
        local_dense_model_path=None,
        local_reranker_model_path=None,
    )

    ft_cfg = FinetuneConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        adapter_mode=args.adapter_mode,
        negatives_per_query=args.negatives_per_query,
        dense_output_dir=args.dense_output_dir,
        reranker_output_dir=args.reranker_output_dir,
        run_dense_finetune=not args.skip_dense,
        run_reranker_finetune=not args.skip_reranker,
    )

    if args.disable_lora:
        ft_cfg.run_dense_finetune = False
        ft_cfg.run_reranker_finetune = False

    corpus_df, corpus_path = build_and_save_corpus(
        config_data_dir=cfg.data_dir,
        artifact_dir=cfg.artifact_dir,
        restrict_to_labeled=cfg.restrict_to_labeled_citations,
        enable_chunking=False,
        chunk_chars=3600,
        overlap_chars=400,
    )
    print(f"Saved corpus for training: {corpus_path}")

    paths = DataPaths(root=Path(cfg.data_dir))
    train_df = pd.read_csv(paths.train)
    val_df = pd.read_csv(paths.val)

    bi_pairs, ce_pairs = build_finetune_pairs(
        train_df=train_df,
        corpus_df=corpus_df,
        dense_model_ref=cfg.dense_model_name,
        negatives_per_query=ft_cfg.negatives_per_query,
        seed=args.seed,
    )
    val_bi_pairs, val_ce_pairs = build_finetune_pairs(
        train_df=val_df,
        corpus_df=corpus_df,
        dense_model_ref=cfg.dense_model_name,
        negatives_per_query=ft_cfg.negatives_per_query,
        seed=args.seed + 1,
    )

    print(f"Train bi-pairs: {len(bi_pairs):,}, train ce-pairs: {len(ce_pairs):,}")
    print(f"Val bi-pairs  : {len(val_bi_pairs):,}, val ce-pairs  : {len(val_ce_pairs):,}")

    if ft_cfg.run_dense_finetune:
        dense_out = finetune_dense(
            bi_pairs=bi_pairs,
            val_bi_pairs=val_bi_pairs,
            dense_model_ref=cfg.dense_model_name,
            ft_cfg=ft_cfg,
            seed=args.seed,
        )
        print(f"Saved dense model to: {dense_out}")

    if ft_cfg.run_reranker_finetune:
        reranker_out = finetune_reranker(
            ce_pairs=ce_pairs,
            val_ce_pairs=val_ce_pairs,
            reranker_model_ref=cfg.reranker_model_name,
            ft_cfg=ft_cfg,
            seed=args.seed,
        )
        print(f"Saved reranker model to: {reranker_out}")


if __name__ == "__main__":
    main()
