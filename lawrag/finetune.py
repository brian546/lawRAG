import random
from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

from .common import normalize_text, split_citations


@dataclass
class FinetuneConfig:
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value", "dense"]
    )
    negatives_per_query: int = 3
    dense_output_dir: str = "artifacts/models/dense_ft"
    reranker_output_dir: str = "artifacts/models/reranker_ft"
    run_dense_finetune: bool = True
    run_reranker_finetune: bool = True


def build_finetune_pairs(
    train_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    negatives_per_query: int = 3,
    seed: int = 42,
) -> Tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    citation_to_text = dict(zip(corpus_df["citation"].tolist(), corpus_df["full_text"].tolist()))
    all_citations = corpus_df["citation"].tolist()

    bi_encoder_pairs: list[dict] = []
    cross_encoder_pairs: list[dict] = []

    for row in train_df.itertuples(index=False):
        query = normalize_text(getattr(row, "query", ""))
        gold_cits = split_citations(getattr(row, "gold_citations", ""))

        positives = [citation_to_text[cit] for cit in gold_cits if cit in citation_to_text]
        if not positives:
            continue

        for pos_text in positives:
            bi_encoder_pairs.append(
                {"anchor": f"query: {query}", "positive": f"passage: {pos_text}"}
            )

        gold_set = set(gold_cits)
        neg_pool = [citation for citation in all_citations if citation not in gold_set]
        neg_sample = rng.sample(neg_pool, min(negatives_per_query, len(neg_pool)))

        for pos_text in positives:
            cross_encoder_pairs.append(
                {"sentence1": query, "sentence2": pos_text, "label": 1.0}
            )
        for neg_cit in neg_sample:
            neg_text = citation_to_text.get(neg_cit, "")
            if neg_text:
                cross_encoder_pairs.append(
                    {"sentence1": query, "sentence2": neg_text, "label": 0.0}
                )

    return bi_encoder_pairs, cross_encoder_pairs


def run_dense_finetuning(
    cfg: FinetuneConfig,
    dense_model_ref: str,
    train_pairs: list[dict],
    val_pairs: list[dict],
    seed: int,
) -> str:
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, TaskType, get_peft_model
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    bi_model = SentenceTransformer(dense_model_ref)

    if cfg.use_lora:
        transformer_module = bi_model._first_module()
        base_auto_model = transformer_module.auto_model
        bi_lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.lora_bias,
            target_modules=cfg.lora_target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        transformer_module.auto_model = get_peft_model(base_auto_model, bi_lora_cfg)

    bi_dataset = HFDataset.from_list(train_pairs)
    bi_eval_dataset = HFDataset.from_list(val_pairs)
    bi_loss = MultipleNegativesRankingLoss(bi_model)

    bi_args = SentenceTransformerTrainingArguments(
        output_dir=cfg.dense_output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=False,
        logging_steps=50,
        seed=seed,
    )

    bi_trainer = SentenceTransformerTrainer(
        model=bi_model,
        args=bi_args,
        train_dataset=bi_dataset,
        eval_dataset=bi_eval_dataset,
        loss=bi_loss,
    )

    bi_trainer.train()
    bi_model.save_pretrained(cfg.dense_output_dir)
    return cfg.dense_output_dir


def run_reranker_finetuning(
    cfg: FinetuneConfig,
    reranker_model_ref: str,
    train_pairs: list[dict],
    val_pairs: list[dict],
    seed: int,
) -> str:
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, TaskType, get_peft_model
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    ce_model = CrossEncoder(reranker_model_ref, num_labels=1)

    if cfg.use_lora:
        ce_lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.lora_bias,
            target_modules=cfg.lora_target_modules,
            task_type=TaskType.SEQ_CLS,
        )
        ce_model.model = get_peft_model(ce_model.model, ce_lora_cfg)

    ce_dataset = HFDataset.from_list(train_pairs)
    ce_eval_dataset = HFDataset.from_list(val_pairs)
    ce_loss = BinaryCrossEntropyLoss(ce_model)

    ce_args = CrossEncoderTrainingArguments(
        output_dir=cfg.reranker_output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=False,
        logging_steps=50,
        seed=seed,
    )

    ce_trainer = CrossEncoderTrainer(
        model=ce_model,
        args=ce_args,
        train_dataset=ce_dataset,
        eval_dataset=ce_eval_dataset,
        loss=ce_loss,
    )

    ce_trainer.train()
    ce_model.save_pretrained(cfg.reranker_output_dir)
    return cfg.reranker_output_dir
