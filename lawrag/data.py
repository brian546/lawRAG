from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import uuid4

import pandas as pd
from tqdm import tqdm

from .common import DataPaths, normalize_text, split_citations


def _ensure_columns(df: pd.DataFrame, defaults: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, default_val in defaults.items():
        if col not in out.columns:
            out[col] = default_val
    return out


def load_labeled_citation_set(paths: DataPaths) -> Set[str]:
    labeled: set[str] = set()
    for csv_path in [paths.train, paths.val]:
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, usecols=["gold_citations"], dtype=str)
        for raw in frame["gold_citations"].fillna(""):
            labeled.update(split_citations(raw))
    return labeled


def build_unified_corpus(
    paths: DataPaths,
    allowed_citations: Optional[Set[str]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def consume_frame(df: pd.DataFrame, source: str) -> None:
        local = _ensure_columns(df, {"title": ""})
        for row in tqdm(local.itertuples(index=False), desc=f"Processing {source}", total=len(local)):
            citation = normalize_text(getattr(row, "citation", ""))
            if not citation:
                continue
            if allowed_citations is not None and citation not in allowed_citations:
                continue

            text = normalize_text(getattr(row, "text", ""))
            title = normalize_text(getattr(row, "title", ""))
            full_text = normalize_text(f"{title}\n{text}")
            rows.append(
                {
                    "doc_id": str(uuid4()),
                    "citation": citation,
                    "text": text,
                    "title": title,
                    "source": source,
                    "full_text": full_text,
                    "length": len(full_text),
                }
            )

    laws = pd.read_csv(paths.laws, usecols=["citation", "text", "title"], dtype=str)
    consume_frame(laws, source="law")

    for chunk in pd.read_csv(paths.court, usecols=["citation", "text"], dtype=str, chunksize=200_000):
        consume_frame(chunk, source="court")

    corpus = pd.DataFrame(rows)
    if corpus.empty:
        return pd.DataFrame(columns=["doc_id", "citation", "text", "title", "source", "full_text", "length"])
    return corpus.reset_index(drop=True)


def chunk_corpus(
    corpus_df: pd.DataFrame,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> pd.DataFrame:
    rows: list[dict] = []
    stride = max(1, chunk_chars - overlap_chars)

    for rec in corpus_df.to_dict("records"):
        text = str(rec["full_text"])
        if len(text) <= chunk_chars:
            rows.append(rec)
            continue

        for start in range(0, len(text), stride):
            piece = text[start : start + chunk_chars]
            new_row = dict(rec)
            new_row["doc_id"] = str(uuid4())
            new_row["full_text"] = piece
            new_row["length"] = len(piece)
            rows.append(new_row)

    return pd.DataFrame(rows)


def build_and_save_corpus(
    config_data_dir: str,
    artifact_dir: str,
    restrict_to_labeled: bool,
    enable_chunking: bool,
    chunk_chars: int,
    overlap_chars: int,
) -> tuple[pd.DataFrame, Path]:
    paths = DataPaths(root=Path(config_data_dir))
    allowed = load_labeled_citation_set(paths) if restrict_to_labeled else None
    corpus_df = build_unified_corpus(paths, allowed_citations=allowed)

    if enable_chunking:
        corpus_df = chunk_corpus(corpus_df, chunk_chars=chunk_chars, overlap_chars=overlap_chars)

    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("corpus_labeled.parquet" if restrict_to_labeled else "corpus.parquet")
    corpus_df.to_parquet(out_path, index=False)
    return corpus_df, out_path
