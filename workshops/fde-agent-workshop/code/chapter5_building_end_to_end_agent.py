"""Chapter 5: Minimal end-to-end agent flow."""


def retrieve_context(query: str) -> str:
    knowledge = {
        "rag": "RAG quality depends on retrieval relevance and chunk quality.",
        "latency": "Latency can be reduced with caching and prompt minimization.",
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return "No high-confidence retrieval context found."


def synthesize_answer(query: str, context: str) -> str:
    return f"Question: {query}\nContext: {context}\nAnswer: Start with measured diagnostics and safe fallbacks."


def run_agent(query: str) -> str:
    context = retrieve_context(query)
    return synthesize_answer(query, context)


if __name__ == "__main__":
    print(run_agent("How do I debug rag quality drops?"))
