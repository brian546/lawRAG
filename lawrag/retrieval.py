from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


@dataclass
class RetrievalResult:
    citation: str
    doc_id: str
    score: float
    source: str
    text: str


def _is_local_adapter_dir(model_name_or_path: str) -> bool:
    model_path = Path(model_name_or_path)
    return model_path.is_dir() and (model_path / "adapter_config.json").exists()


def _get_base_model_from_adapter(adapter_dir: str) -> Optional[str]:
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.exists():
        return None
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    base_ref = cfg.get("base_model_name_or_path")
    return str(base_ref) if base_ref else None


def load_sentence_transformer(model_name_or_path: str):
    from sentence_transformers import SentenceTransformer

    default_model = lambda: SentenceTransformer(model_name_or_path, trust_remote_code=True)

    if not _is_local_adapter_dir(model_name_or_path):
        return default_model()

    base_ref = _get_base_model_from_adapter(model_name_or_path)
    if not base_ref:
        return default_model()

    from peft import PeftModel

    model = SentenceTransformer(base_ref, trust_remote_code=True)
    auto_model = model._first_module().auto_model  # pyright: ignore[reportPrivateUsage]

    peft_model = PeftModel.from_pretrained(auto_model, model_name_or_path)
    if hasattr(peft_model, "merge_and_unload"):
        _ = peft_model.merge_and_unload()
    return model


def load_cross_encoder(model_name_or_path: str):
    from sentence_transformers import CrossEncoder

    if not _is_local_adapter_dir(model_name_or_path):
        return CrossEncoder(model_name_or_path)

    base_ref = _get_base_model_from_adapter(model_name_or_path)
    if not base_ref:
        return CrossEncoder(model_name_or_path)

    from peft import PeftModel

    model = CrossEncoder(base_ref, num_labels=1)
    peft_model = PeftModel.from_pretrained(model.model, model_name_or_path)
    if hasattr(peft_model, "merge_and_unload"):
        model.model = peft_model.merge_and_unload()
    else:
        model.model = peft_model
    return model


class FaissCitationRetriever:
    class _SentenceTransformerEmbeddings(Embeddings):
        def __init__(self, model_name_or_path: str):
            self.model = load_sentence_transformer(model_name_or_path)

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return emb.tolist()

        def embed_query(self, text: str) -> List[float]:
            emb = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)
            return emb[0].tolist()

    def __init__(self, corpus_df: pd.DataFrame, model_name_or_path: str):
        from langchain_community.vectorstores import FAISS

        self.corpus = corpus_df.reset_index(drop=True)
        self.embedding = self._SentenceTransformerEmbeddings(model_name_or_path)

        docs = [
            Document(
                page_content=row.get("full_text", "") or "",
                metadata={
                    "citation": row["citation"],
                    "doc_id": row["doc_id"],
                    "source": row.get("source", ""),
                },
            )
            for row in self.corpus.to_dict("records")
        ]

        self.vectorstore = FAISS.from_documents(documents=docs, embedding=self.embedding)

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        return 1.0 / (1.0 + max(float(distance), 0.0))

    def search(self, query: str, topk: int = 20) -> List[RetrievalResult]:
        out: list[RetrievalResult] = []
        docs_and_dist = self.vectorstore.similarity_search_with_score(query, k=topk)
        for doc, dist in docs_and_dist:
            out.append(
                RetrievalResult(
                    citation=doc.metadata.get("citation", ""),
                    doc_id=doc.metadata.get("doc_id", ""),
                    score=self._distance_to_similarity(float(dist)),
                    source=doc.metadata.get("source", ""),
                    text=doc.page_content,
                )
            )
        return out


def aggregate_results_by_citation(results: List[RetrievalResult], topk: Optional[int] = None) -> List[RetrievalResult]:
    grouped: Dict[str, List[RetrievalResult]] = {}
    for rec in results:
        grouped.setdefault(rec.citation, []).append(rec)

    aggregated: list[RetrievalResult] = []
    for citation, recs in grouped.items():
        max_text_rec = max(recs, key=lambda r: r.score)
        aggregated.append(
            RetrievalResult(
                citation=citation,
                doc_id=max_text_rec.doc_id,
                score=float(max_text_rec.score),
                source=max_text_rec.source,
                text=max_text_rec.text,
            )
        )

    aggregated.sort(key=lambda r: r.score, reverse=True)
    return aggregated if topk is None else aggregated[:topk]


class CrossEncoderCitationReranker:
    def __init__(self, model_name_or_path: str):
        self.model = load_cross_encoder(model_name_or_path)

    def rerank(self, query: str, candidates: List[RetrievalResult], batch_size: int = 32) -> List[RetrievalResult]:
        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        rescored: list[RetrievalResult] = []
        for candidate, score in zip(candidates, scores):
            rescored.append(
                RetrievalResult(
                    citation=candidate.citation,
                    doc_id=candidate.doc_id,
                    score=float(score),
                    source=candidate.source,
                    text=candidate.text,
                )
            )

        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored
