from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

SEED = 42
MULTISPACE_RE = re.compile(r"\s+")


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return MULTISPACE_RE.sub(" ", str(text)).strip()


def split_citations(raw: str | None) -> List[str]:
    if raw is None:
        return []
    parts = [normalize_text(part) for part in str(raw).split(";")]
    return [part for part in parts if part]


def join_citations(citations: Sequence[str]) -> str:
    seen = set()
    out: list[str] = []
    for citation in citations:
        c = normalize_text(citation)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return ";".join(out)


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


@dataclass
class Config:
    data_dir: str = "data"
    artifact_dir: str = "artifacts"
    submission_name: str = "submission.csv"

    restrict_to_labeled_citations: bool = True
    enable_chunking: bool = False
    chunk_chars: int = 3600
    overlap_chars: int = 400

    enable_finetuning: bool = True

    mode: str = "dense"
    out_k: int = 20
    stage_k: int = 150

    use_reranker: bool = True
    reranker_top_n: int = 50
    reranker_batch_size: int = 32

    threshold: Optional[float] = 0.5

    dense_model_name: str = "intfloat/multilingual-e5-base"
    reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    local_dense_model_path: Optional[str] = None
    local_reranker_model_path: Optional[str] = None

    enable_mlflow: bool = False
    mlflow_tracking_uri: Optional[str] = "http://127.0.0.1:5000"
    mlflow_experiment: str = "lawrag-dense"


@dataclass
class FinetuneConfig:
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1

    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2

    adapter_mode: str = "lora"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: List[str] | str = field(default_factory=lambda: "all-linear")

    qlora_4bit_quant_type: str = "nf4"
    qlora_use_double_quant: bool = True

    negatives_per_query: int = 3

    dense_output_dir: str = "artifacts/models/dense_ft"
    reranker_output_dir: str = "artifacts/models/reranker_ft"

    run_dense_finetune: bool = True
    run_reranker_finetune: bool = True


@dataclass
class PipelineConfig:
    mode: str = "dense"
    dense_model_name: str = "intfloat/multilingual-e5-base"
    stage_k: int = 150
    out_k: int = 12
    use_reranker: bool = True
    reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_top_n: int = 50
    reranker_batch_size: int = 32
    threshold: float | None = 0.5


@dataclass
class DataPaths:
    root: Path

    @property
    def train(self) -> Path:
        return self.root / "train.csv"

    @property
    def val(self) -> Path:
        return self.root / "val.csv"

    @property
    def test(self) -> Path:
        return self.root / "test.csv"

    @property
    def laws(self) -> Path:
        laws_unique = self.root / "laws_de_unique.csv"
        return laws_unique if laws_unique.exists() else (self.root / "laws_de.csv")

    @property
    def court(self) -> Path:
        court_unique = self.root / "court_considerations_unique.csv"
        return court_unique if court_unique.exists() else (self.root / "court_considerations.csv")
