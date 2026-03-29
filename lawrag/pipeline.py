from __future__ import annotations

from typing import List, Sequence, Set, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from .common import PipelineConfig, join_citations, split_citations, unique_preserve_order
from .retrieval import (
    CrossEncoderCitationReranker,
    FaissCitationRetriever,
    RetrievalResult,
    aggregate_results_by_citation,
)


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
        local_dense_model_path: str | None = None,
        local_reranker_model_path: str | None = None,
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
        graph.add_node("rerank", self._rerank)
        graph.add_node("filter", self._filter_by_threshold)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "retrieve_stage")
        graph.add_edge("retrieve_stage", "aggregate")
        graph.add_conditional_edges("aggregate", self._determine_rerank, {"rerank": "rerank", "no_rerank": "filter"})
        graph.add_edge("rerank", "filter")
        graph.add_edge("filter", "finalize")
        graph.add_edge("finalize", END)
        self.graph = graph.compile()

    def _retrieve_stage(self, state: PipelineState) -> PipelineState:
        query = state["query"]
        stage_results = self.dense.search(query, topk=self.config.stage_k)
        return {"stage_results": stage_results}

    def _aggregate(self, state: PipelineState) -> PipelineState:
        stage_results = state.get("stage_results", [])
        aggregated = aggregate_results_by_citation(stage_results, topk=self.config.stage_k)
        return {"aggregated": aggregated}

    def _determine_rerank(self, state: PipelineState) -> str:
        _ = state
        if self.config.use_reranker and self.reranker is not None:
            return "rerank"
        return "no_rerank"

    def _rerank(self, state: PipelineState) -> PipelineState:
        aggregated = state.get("aggregated", [])
        query = state["query"]
        top_n = min(self.config.reranker_top_n, len(aggregated))
        head = aggregated[:top_n]
        reranked_head = self.reranker.rerank(query, head, batch_size=self.config.reranker_batch_size)
        sorted_reranked = sorted(reranked_head, key=lambda r: r.score, reverse=True)
        return {"aggregated": sorted_reranked}

    def _filter_by_threshold(self, state: PipelineState) -> PipelineState:
        aggregated = state.get("aggregated", [])

        threshold = self.config.threshold if self.config.threshold is not None else 0.0
        if threshold > 0.0:
            logit_threshold = np.log(threshold / (1.0 - threshold)) if threshold != 0.5 else 0.0
            if aggregated and aggregated[0].score >= logit_threshold:
                filtered = [r for r in aggregated if r.score >= logit_threshold]
            else:
                filtered = aggregated[:1]
        else:
            filtered = aggregated
        return {"aggregated": filtered}

    def _finalize(self, state: PipelineState) -> PipelineState:
        aggregated = state["aggregated"]
        return {"final_results": aggregated[: self.config.out_k]}

    def retrieve(self, query: str) -> List[RetrievalResult]:
        out = self.graph.invoke({"query": query})
        return out.get("final_results", [])

    def predict(self, query: str) -> str:
        results = self.retrieve(query)
        citations = unique_preserve_order([r.citation for r in results])
        return join_citations(citations[: self.config.out_k])


def precision_for_sets(gold: Set[str], pred: Set[str]) -> float:
    if not pred:
        return 1.0 if not gold else 0.0
    tp = len(gold & pred)
    return tp / max(len(pred), 1)


def recall_for_sets(gold: Set[str], pred: Set[str]) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    tp = len(gold & pred)
    return tp / max(len(gold), 1)


def f1_for_sets(gold: Set[str], pred: Set[str]) -> float:
    if not gold and not pred:
        return 1.0
    if not pred:
        return 0.0
    precision = precision_for_sets(gold, pred)
    recall = recall_for_sets(gold, pred)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_precision(gold_list: Sequence[str], pred_list: Sequence[str]) -> float:
    assert len(gold_list) == len(pred_list)
    scores = []
    for g_raw, p_raw in zip(gold_list, pred_list):
        g = set(split_citations(g_raw))
        p = set(split_citations(p_raw))
        scores.append(precision_for_sets(g, p))
    return float(sum(scores) / len(scores)) if scores else 0.0


def macro_recall(gold_list: Sequence[str], pred_list: Sequence[str]) -> float:
    assert len(gold_list) == len(pred_list)
    scores = []
    for g_raw, p_raw in zip(gold_list, pred_list):
        g = set(split_citations(g_raw))
        p = set(split_citations(p_raw))
        scores.append(recall_for_sets(g, p))
    return float(sum(scores) / len(scores)) if scores else 0.0


def macro_f1(gold_list: Sequence[str], pred_list: Sequence[str]) -> float:
    assert len(gold_list) == len(pred_list)
    scores = []
    for g_raw, p_raw in zip(gold_list, pred_list):
        g = set(split_citations(g_raw))
        p = set(split_citations(p_raw))
        scores.append(f1_for_sets(g, p))
    return float(sum(scores) / len(scores)) if scores else 0.0
