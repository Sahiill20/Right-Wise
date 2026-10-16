from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker() -> CrossEncoder:
    """Load the cross-encoder reranking model."""
    print(f"Loading reranker: {MODEL_NAME}")
    return CrossEncoder(MODEL_NAME)


def rerank(
    query: str,
    candidates: list[dict],
    model: CrossEncoder,
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank candidate chunks using a cross-encoder.

    The model receives (query, candidate_text) pairs together.
    """

    pairs = [
        (query, candidate["text"])
        for candidate in candidates
    ]

    scores = model.predict(pairs)

    reranked = []

    for candidate, score in zip(candidates, scores):
        result = candidate.copy()
        result["reranker_score"] = float(score)
        reranked.append(result)

    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True,
    )

    return reranked[:top_k]