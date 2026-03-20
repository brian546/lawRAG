from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

from .common import normalize_text, split_citations


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
        return self.root / "laws_de.csv"

    @property
    def court(self) -> Path:
        return self.root / "court_considerations.csv"


def ensure_columns(df: pd.DataFrame, defaults: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, default_value in defaults.items():
        if col not in out.columns:
            out[col] = default_value
    return out


def load_labeled_citation_set(paths: DataPaths) -> Set[str]:
    labeled: Set[str] = set()
    for file_path in [paths.train, paths.val]:
        if not file_path.exists():
            continue
        df = pd.read_csv(file_path, usecols=["gold_citations"], dtype=str)
        for raw in df["gold_citations"].fillna(""):
            labeled.update(split_citations(raw))
    return labeled


def build_unified_corpus(
    paths: DataPaths,
    allowed_citations: Optional[Set[str]] = None,
) -> pd.DataFrame:
    best_by_citation: Dict[str, Dict[str, object]] = {}

    def consume_frame(df: pd.DataFrame, source: str) -> None:
        local = ensure_columns(df, {"title": ""})
        for row in local.itertuples(index=False):
            citation = normalize_text(getattr(row, "citation", ""))
            if not citation:
                continue
            if allowed_citations is not None and citation not in allowed_citations:
                continue

            text = normalize_text(getattr(row, "text", ""))
            title = normalize_text(getattr(row, "title", ""))
            full_text = normalize_text(f"{title}\n{text}")
            length = len(full_text)

            prev = best_by_citation.get(citation)
            if prev is None or length > int(prev["length"]):
                best_by_citation[citation] = {
                    "doc_id": f"{citation}__chunk0",
                    "citation": citation,
                    "text": text,
                    "title": title,
                    "source": source,
                    "full_text": full_text,
                    "length": length,
                }

    laws = pd.read_csv(paths.laws, usecols=["citation", "text", "title"], dtype=str)
    consume_frame(laws, source="law")

    for chunk in pd.read_csv(
        paths.court,
        usecols=["citation", "text"],
        dtype=str,
        chunksize=200_000,
    ):
        consume_frame(chunk, source="court")

    corpus = pd.DataFrame(best_by_citation.values()).sort_values("citation")
    return corpus.reset_index(drop=True)


def chunk_corpus(
    corpus_df: pd.DataFrame,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> pd.DataFrame:
    rows = []
    stride = max(1, chunk_chars - overlap_chars)

    for rec in corpus_df.to_dict("records"):
        text = str(rec["full_text"])
        if len(text) <= chunk_chars:
            rows.append(rec)
            continue

        chunk_idx = 0
        for start in range(0, len(text), stride):
            piece = text[start : start + chunk_chars]
            if len(piece) < 80:
                continue
            new_row = dict(rec)
            new_row["doc_id"] = f"{rec['citation']}__chunk{chunk_idx}"
            new_row["full_text"] = piece
            new_row["length"] = len(piece)
            rows.append(new_row)
            chunk_idx += 1

    return pd.DataFrame(rows)
