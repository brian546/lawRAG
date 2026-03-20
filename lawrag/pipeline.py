from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from .common import join_citations, split_citations, unique_preserve_order
from .retrieval import (
    CrossEncoderCitationReranker,
    FaissCitationRetriever,
    RetrievalResult,
    aggregate_results_by_citation,
)


@dataclass
class PipelineConfig:
    mode: str = "dense"
    dense_model_name: str = "intfloat/multilingual-e5-base"
    citation_score_agg: str = "max"
    stage_k: int = 150
    out_k: int = 12
    use_reranker: bool = True
    reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_top_n: int = 50
    reranker_batch_size: int = 32


class PipelineState(TypedDict, total=False):
    query: str
    stage_results: List[RetrievalResult]
    aggregated: List[RetrievalResult]
    final_results: List[RetrievalResult]


class CitationRAGPipeline:
    def __init__(
        self,
        corpus_df: pd.DataFrame,
        config: PipelineConfig,
        local_dense_model_path: Optional[str] = None,
        local_reranker_model_path: Optional[str] = None,
    ):
        self.config = config

        dense_model_ref = local_dense_model_path or self.config.dense_model_name
        reranker_model_ref = local_reranker_model_path or self.config.reranker_model_name

        self.dense = FaissCitationRetriever(corpus_df, dense_model_ref)

        self.reranker = None
        if self.config.use_reranker:
            self.reranker = CrossEncoderCitationReranker(reranker_model_ref)

        graph = StateGraph(PipelineState)
        graph.add_node("retrieve_stage", self._retrieve_stage)
        graph.add_node("aggregate", self._aggregate)
        graph.add_node("rerank_or_finalize", self._rerank_or_finalize)
        graph.add_edge(START, "retrieve_stage")
        graph.add_edge("retrieve_stage", "aggregate")
        graph.add_edge("aggregate", "rerank_or_finalize")
        graph.add_edge("rerank_or_finalize", END)
        self.graph = graph.compile()

    def _retrieve_stage(self, state: PipelineState) -> PipelineState:
        query = state["query"]
        stage_results = self.dense.search(query, topk=self.config.stage_k)
        return {"stage_results": stage_results}

    def _aggregate(self, state: PipelineState) -> PipelineState:
        aggregated = aggregate_results_by_citation(
            state.get("stage_results", []),
            score_agg=self.config.citation_score_agg,
            topk=self.config.stage_k,
        )
        return {"aggregated": aggregated}

    def _rerank_or_finalize(self, state: PipelineState) -> PipelineState:
        aggregated = state.get("aggregated", [])
        if self.reranker is None:
            return {"final_results": aggregated[: self.config.out_k]}

        query = state["query"]
        top_n = min(self.config.reranker_top_n, len(aggregated))
        head = aggregated[:top_n]
        tail = aggregated[top_n:]
        reranked_head = self.reranker.rerank(
            query,
            head,
            batch_size=self.config.reranker_batch_size,
        )
        return {"final_results": (reranked_head + tail)[: self.config.out_k]}

    def retrieve(self, query: str) -> List[RetrievalResult]:
        out = self.graph.invoke({"query": query})
        return out.get("final_results", [])

    def predict(self, query: str) -> str:
        results = self.retrieve(query)
        citations = unique_preserve_order([rec.citation for rec in results])
        return join_citations(citations[: self.config.out_k])


def f1_for_sets(gold: Set[str], pred: Set[str]) -> float:
    if not gold and not pred:
        return 1.0
    if not pred:
        return 0.0

    tp = len(gold & pred)
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(gold), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(gold_list: Sequence[str], pred_list: Sequence[str]) -> float:
    if len(gold_list) != len(pred_list):
        raise ValueError("gold_list and pred_list must have same length")

    scores = []
    for g_raw, p_raw in zip(gold_list, pred_list):
        gold = set(split_citations(g_raw))
        pred = set(split_citations(p_raw))
        scores.append(f1_for_sets(gold, pred))
    return float(sum(scores) / len(scores)) if scores else 0.0
