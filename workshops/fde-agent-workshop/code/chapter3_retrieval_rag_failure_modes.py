"""Chapter 3: Simple retrieval quality diagnostics."""

from dataclasses import dataclass


@dataclass
class RetrievalStats:
    queries: int
    with_relevant_doc: int
    false_positive_docs: int


def recall_at_k(stats: RetrievalStats) -> float:
    if stats.queries == 0:
        return 0.0
    return stats.with_relevant_doc / stats.queries


def precision_proxy(stats: RetrievalStats, avg_docs_per_query: int) -> float:
    if stats.queries == 0 or avg_docs_per_query <= 0:
        return 0.0
    total_docs = stats.queries * avg_docs_per_query
    true_positive = max(0, stats.with_relevant_doc)
    return max(0.0, min(1.0, true_positive / total_docs))


if __name__ == "__main__":
    sample = RetrievalStats(queries=100, with_relevant_doc=72, false_positive_docs=55)
    print("Recall@k:", round(recall_at_k(sample), 3))
    print("Precision proxy:", round(precision_proxy(sample, avg_docs_per_query=3), 3))
