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
    FinetuneConfig,
    get_config_section,
    load_json_config,
    set_global_seed,
)
from lawrag.data import build_and_save_corpus
from lawrag.finetune import build_finetune_pairs, finetune_dense, finetune_reranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LawRAG dense retriever and reranker.")
    parser.add_argument("--config", default=None, help="Path to JSON config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shared_config = get_config_section(load_json_config(args.config), "shared")
    fine_tune_config = get_config_section(load_json_config(args.config), "fine_tune")

    seed = int(shared_config.get("seed", SEED))
    set_global_seed(seed)

    cfg = Config(
        data_dir=shared_config.get("data_dir", "data"),
        artifact_dir=shared_config.get("artifact_dir", "artifacts"),
        restrict_to_labeled_citations=fine_tune_config.get("restrict_to_labeled", False),
        enable_chunking=fine_tune_config.get("enable_chunking", False),
        chunk_chars=fine_tune_config.get("chunk_chars", 3600),
        overlap_chars=fine_tune_config.get("overlap_chars", 400),
        dense_model_name=fine_tune_config.get("dense_model_name", "intfloat/multilingual-e5-base"),
        reranker_model_name=fine_tune_config.get("reranker_model_name", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
        local_dense_model_path=None,
        local_reranker_model_path=None,
    )

    ft_cfg = FinetuneConfig(
        epochs=fine_tune_config.get("epochs", 3),
        batch_size=fine_tune_config.get("batch_size", 16),
        learning_rate=fine_tune_config.get("learning_rate", 2e-5),
        warmup_ratio=fine_tune_config.get("warmup_ratio", 0.1),
        gradient_accumulation_steps=fine_tune_config.get("gradient_accumulation_steps", 1),
        eval_steps=fine_tune_config.get("eval_steps", 100),
        save_steps=fine_tune_config.get("save_steps", 100),
        save_total_limit=fine_tune_config.get("save_total_limit", 2),
        adapter_mode=fine_tune_config.get("adapter_mode", "lora"),
        lora_r=fine_tune_config.get("lora_r", 16),
        lora_alpha=fine_tune_config.get("lora_alpha", 32),
        lora_dropout=fine_tune_config.get("lora_dropout", 0.05),
        lora_bias=fine_tune_config.get("lora_bias", "none"),
        lora_target_modules=fine_tune_config.get("lora_target_modules", "all-linear"),
        qlora_4bit_quant_type=fine_tune_config.get("qlora_4bit_quant_type", "nf4"),
        qlora_use_double_quant=fine_tune_config.get("qlora_use_double_quant", True),
        negatives_per_query=fine_tune_config.get("negatives_per_query", 3),
        dense_output_dir=fine_tune_config.get("dense_output_dir", "artifacts/models/dense_ft"),
        reranker_output_dir=fine_tune_config.get("reranker_output_dir", "artifacts/models/reranker_ft"),
        run_dense_finetune=fine_tune_config.get("run_dense_finetune", True),
        run_reranker_finetune=fine_tune_config.get("run_reranker_finetune", True),
    )

    skip_dense = fine_tune_config.get("skip_dense", False)
    skip_reranker = fine_tune_config.get("skip_reranker", False)
    disable_lora = fine_tune_config.get("disable_lora", False)

    if bool(skip_dense):
        ft_cfg.run_dense_finetune = False
    if bool(skip_reranker):
        ft_cfg.run_reranker_finetune = False
    if bool(disable_lora):
        ft_cfg.run_dense_finetune = False
        ft_cfg.run_reranker_finetune = False

    corpus_df, corpus_path = build_and_save_corpus(
        config_data_dir=cfg.data_dir,
        artifact_dir=cfg.artifact_dir,
        restrict_to_labeled=cfg.restrict_to_labeled_citations,
        enable_chunking=cfg.enable_chunking,
        chunk_chars=cfg.chunk_chars,
        overlap_chars=cfg.overlap_chars,
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
        seed=seed,
    )
    val_bi_pairs, val_ce_pairs = build_finetune_pairs(
        train_df=val_df,
        corpus_df=corpus_df,
        dense_model_ref=cfg.dense_model_name,
        negatives_per_query=ft_cfg.negatives_per_query,
        seed=seed + 1,
    )

    print(f"Train bi-pairs: {len(bi_pairs):,}, train ce-pairs: {len(ce_pairs):,}")
    print(f"Val bi-pairs  : {len(val_bi_pairs):,}, val ce-pairs  : {len(val_ce_pairs):,}")

    if ft_cfg.run_dense_finetune:
        dense_out = finetune_dense(
            bi_pairs=bi_pairs,
            val_bi_pairs=val_bi_pairs,
            dense_model_ref=cfg.dense_model_name,
            ft_cfg=ft_cfg,
            seed=seed,
        )
        print(f"Saved dense model to: {dense_out}")

    if ft_cfg.run_reranker_finetune:
        reranker_out = finetune_reranker(
            ce_pairs=ce_pairs,
            val_ce_pairs=val_ce_pairs,
            reranker_model_ref=cfg.reranker_model_name,
            ft_cfg=ft_cfg,
            seed=seed,
        )
        print(f"Saved reranker model to: {reranker_out}")


if __name__ == "__main__":
    main()
