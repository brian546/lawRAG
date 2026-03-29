from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from .common import FinetuneConfig, SEED, normalize_text, split_citations


def build_finetune_pairs(
    train_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    dense_model_ref: str,
    negatives_per_query: int = 3,
    seed: int = SEED,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    citation_to_text: dict[str, str] = dict(zip(corpus_df["citation"].tolist(), corpus_df["full_text"].tolist()))
    all_citations = corpus_df["citation"].tolist()

    dense_ref = str(dense_model_ref).lower()
    use_e5_prefix = "e5" in dense_ref

    def format_query_for_dense(query: str) -> str:
        return f"query: {query}" if use_e5_prefix else query

    def format_passage_for_dense(passage: str) -> str:
        return f"passage: {passage}" if use_e5_prefix else passage

    bi_encoder_pairs: list[dict] = []
    cross_encoder_pairs: list[dict] = []

    for row in train_df.itertuples(index=False):
        query = normalize_text(getattr(row, "query", ""))
        gold_cits = split_citations(getattr(row, "gold_citations", ""))

        positives = [citation_to_text[citation] for citation in gold_cits if citation in citation_to_text]
        if not positives:
            continue

        dense_query = format_query_for_dense(query)
        for pos_text in positives:
            bi_encoder_pairs.append({"anchor": dense_query, "positive": format_passage_for_dense(pos_text)})

        gold_set = set(gold_cits)
        neg_pool = [citation for citation in all_citations if citation not in gold_set]
        neg_sample = rng.sample(neg_pool, min(negatives_per_query, len(neg_pool)))

        for pos_text in positives:
            cross_encoder_pairs.append({"sentence1": query, "sentence2": pos_text, "label": 1.0})
        for neg_cit in neg_sample:
            neg_text = citation_to_text.get(neg_cit, "")
            if neg_text:
                cross_encoder_pairs.append({"sentence1": query, "sentence2": neg_text, "label": 0.0})

    return bi_encoder_pairs, cross_encoder_pairs


def finetune_dense(
    bi_pairs: list[dict],
    val_bi_pairs: list[dict],
    dense_model_ref: str,
    ft_cfg: FinetuneConfig,
    seed: int = SEED,
) -> str:
    import torch
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers
    from transformers import BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("LoRA/QLoRA fine-tuning requires a CUDA-enabled GPU.")

    adapter_mode = str(ft_cfg.adapter_mode).lower()
    use_qlora = adapter_mode == "qlora"

    if use_qlora:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=ft_cfg.qlora_4bit_quant_type,
            bnb_4bit_use_double_quant=ft_cfg.qlora_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = SentenceTransformer(
            dense_model_ref,
            model_kwargs={"quantization_config": bnb_cfg, "device_map": "auto"},
            trust_remote_code=True,
        )
        transformer_module = model._first_module()  # pyright: ignore[reportPrivateUsage]
        base_auto_model = prepare_model_for_kbit_training(transformer_module.auto_model)
        use_bf16 = bool(compute_dtype == torch.bfloat16)
        use_fp16 = bool(compute_dtype == torch.float16)
    else:
        model = SentenceTransformer(dense_model_ref, trust_remote_code=True)
        transformer_module = model._first_module()  # pyright: ignore[reportPrivateUsage]
        base_auto_model = transformer_module.auto_model
        use_bf16 = bool(torch.cuda.is_bf16_supported())
        use_fp16 = not use_bf16

    lora_cfg = LoraConfig(
        r=ft_cfg.lora_r,
        lora_alpha=ft_cfg.lora_alpha,
        lora_dropout=ft_cfg.lora_dropout,
        bias=ft_cfg.lora_bias,
        target_modules=ft_cfg.lora_target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    transformer_module.auto_model = get_peft_model(base_auto_model, lora_cfg)

    train_dataset = HFDataset.from_list(bi_pairs)
    eval_dataset = HFDataset.from_list(val_bi_pairs)
    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=ft_cfg.dense_output_dir,
        num_train_epochs=ft_cfg.epochs,
        per_device_train_batch_size=ft_cfg.batch_size,
        gradient_accumulation_steps=ft_cfg.gradient_accumulation_steps,
        learning_rate=ft_cfg.learning_rate,
        warmup_ratio=ft_cfg.warmup_ratio,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="steps",
        save_steps=ft_cfg.save_steps,
        save_total_limit=ft_cfg.save_total_limit,
        eval_strategy="steps",
        eval_steps=ft_cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=50,
        seed=seed,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    trainer.train()

    out_dir = Path(ft_cfg.dense_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    return str(out_dir)


def finetune_reranker(
    ce_pairs: list[dict],
    val_ce_pairs: list[dict],
    reranker_model_ref: str,
    ft_cfg: FinetuneConfig,
    seed: int = SEED,
) -> str:
    import torch
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from transformers import BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("LoRA/QLoRA fine-tuning requires a CUDA-enabled GPU.")

    adapter_mode = str(ft_cfg.adapter_mode).lower()
    use_qlora = adapter_mode == "qlora"

    if use_qlora:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=ft_cfg.qlora_4bit_quant_type,
            bnb_4bit_use_double_quant=ft_cfg.qlora_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = CrossEncoder(
            reranker_model_ref,
            num_labels=1,
            model_kwargs={"quantization_config": bnb_cfg, "device_map": "auto"},
        )
        model.model = prepare_model_for_kbit_training(model.model)
        use_bf16 = bool(compute_dtype == torch.bfloat16)
        use_fp16 = bool(compute_dtype == torch.float16)
    else:
        model = CrossEncoder(reranker_model_ref, num_labels=1)
        use_bf16 = bool(torch.cuda.is_bf16_supported())
        use_fp16 = not use_bf16

    lora_cfg = LoraConfig(
        r=ft_cfg.lora_r,
        lora_alpha=ft_cfg.lora_alpha,
        lora_dropout=ft_cfg.lora_dropout,
        bias=ft_cfg.lora_bias,
        target_modules=ft_cfg.lora_target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model.model = get_peft_model(model.model, lora_cfg)

    train_dataset = HFDataset.from_list(ce_pairs)
    eval_dataset = HFDataset.from_list(val_ce_pairs)
    loss = BinaryCrossEntropyLoss(model)

    args = CrossEncoderTrainingArguments(
        output_dir=ft_cfg.reranker_output_dir,
        num_train_epochs=ft_cfg.epochs,
        per_device_train_batch_size=ft_cfg.batch_size,
        gradient_accumulation_steps=ft_cfg.gradient_accumulation_steps,
        learning_rate=ft_cfg.learning_rate,
        warmup_ratio=ft_cfg.warmup_ratio,
        save_strategy="steps",
        save_steps=ft_cfg.save_steps,
        save_total_limit=ft_cfg.save_total_limit,
        eval_strategy="steps",
        eval_steps=ft_cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=50,
        seed=seed,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    trainer.train()

    out_dir = Path(ft_cfg.reranker_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    return str(out_dir)
