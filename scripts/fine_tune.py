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
    resolve_options,
    set_global_seed,
)
from lawrag.data import build_and_save_corpus
from lawrag.finetune import build_finetune_pairs, finetune_dense, finetune_reranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LawRAG dense retriever and reranker.")
    parser.add_argument("--config", default=None, help="Path to shared JSON config file.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--restrict-to-labeled", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--dense-model-name", default=None)
    parser.add_argument("--reranker-model-name", default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--negatives-per-query", type=int, default=None)

    parser.add_argument("--adapter-mode", choices=["lora", "qlora"], default=None)
    parser.add_argument("--disable-lora", action=argparse.BooleanOptionalAction, default=None, help="Disable all fine-tuning stages.")

    parser.add_argument("--dense-output-dir", default=None)
    parser.add_argument("--reranker-output-dir", default=None)

    parser.add_argument("--skip-dense", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-reranker", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_config = get_config_section(load_json_config(args.config), "fine_tune")
    resolved = resolve_options(
        args,
        file_config,
        [
            "data_dir",
            "artifact_dir",
            "restrict_to_labeled",
            "dense_model_name",
            "reranker_model_name",
            "epochs",
            "batch_size",
            "learning_rate",
            "warmup_ratio",
            "gradient_accumulation_steps",
            "eval_steps",
            "save_steps",
            "save_total_limit",
            "adapter_mode",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "lora_bias",
            "lora_target_modules",
            "qlora_4bit_quant_type",
            "qlora_use_double_quant",
            "negatives_per_query",
            "dense_output_dir",
            "reranker_output_dir",
            "run_dense_finetune",
            "run_reranker_finetune",
            "disable_lora",
            "skip_dense",
            "skip_reranker",
            "enable_chunking",
            "chunk_chars",
            "overlap_chars",
            "seed",
        ],
    )

    seed = int(resolved.get("seed", SEED))
    set_global_seed(seed)

    base_cfg = Config()
    base_ft_cfg = FinetuneConfig()

    cfg = Config(
        data_dir=resolved.get("data_dir", base_cfg.data_dir),
        artifact_dir=resolved.get("artifact_dir", base_cfg.artifact_dir),
        restrict_to_labeled_citations=resolved.get("restrict_to_labeled", base_cfg.restrict_to_labeled_citations),
        enable_chunking=resolved.get("enable_chunking", base_cfg.enable_chunking),
        chunk_chars=resolved.get("chunk_chars", base_cfg.chunk_chars),
        overlap_chars=resolved.get("overlap_chars", base_cfg.overlap_chars),
        dense_model_name=resolved.get("dense_model_name", base_cfg.dense_model_name),
        reranker_model_name=resolved.get("reranker_model_name", base_cfg.reranker_model_name),
        local_dense_model_path=None,
        local_reranker_model_path=None,
    )

    ft_cfg = FinetuneConfig(
        epochs=resolved.get("epochs", base_ft_cfg.epochs),
        batch_size=resolved.get("batch_size", base_ft_cfg.batch_size),
        learning_rate=resolved.get("learning_rate", base_ft_cfg.learning_rate),
        warmup_ratio=resolved.get("warmup_ratio", base_ft_cfg.warmup_ratio),
        gradient_accumulation_steps=resolved.get("gradient_accumulation_steps", base_ft_cfg.gradient_accumulation_steps),
        eval_steps=resolved.get("eval_steps", base_ft_cfg.eval_steps),
        save_steps=resolved.get("save_steps", base_ft_cfg.save_steps),
        save_total_limit=resolved.get("save_total_limit", base_ft_cfg.save_total_limit),
        adapter_mode=resolved.get("adapter_mode", base_ft_cfg.adapter_mode),
        lora_r=resolved.get("lora_r", base_ft_cfg.lora_r),
        lora_alpha=resolved.get("lora_alpha", base_ft_cfg.lora_alpha),
        lora_dropout=resolved.get("lora_dropout", base_ft_cfg.lora_dropout),
        lora_bias=resolved.get("lora_bias", base_ft_cfg.lora_bias),
        lora_target_modules=resolved.get("lora_target_modules", base_ft_cfg.lora_target_modules),
        qlora_4bit_quant_type=resolved.get("qlora_4bit_quant_type", base_ft_cfg.qlora_4bit_quant_type),
        qlora_use_double_quant=resolved.get("qlora_use_double_quant", base_ft_cfg.qlora_use_double_quant),
        negatives_per_query=resolved.get("negatives_per_query", base_ft_cfg.negatives_per_query),
        dense_output_dir=resolved.get("dense_output_dir", base_ft_cfg.dense_output_dir),
        reranker_output_dir=resolved.get("reranker_output_dir", base_ft_cfg.reranker_output_dir),
        run_dense_finetune=resolved.get("run_dense_finetune", base_ft_cfg.run_dense_finetune),
        run_reranker_finetune=resolved.get("run_reranker_finetune", base_ft_cfg.run_reranker_finetune),
    )

    if bool(resolved.get("skip_dense", False)):
        ft_cfg.run_dense_finetune = False
    if bool(resolved.get("skip_reranker", False)):
        ft_cfg.run_reranker_finetune = False
    if bool(resolved.get("disable_lora", False)):
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
