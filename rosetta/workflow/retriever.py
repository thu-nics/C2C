"""Local search engine retriever for HotpotQA articles."""

from typing import Any

from camel.storages import QdrantStorage, VectorDBQuery

from rosetta.workflow.embeddings import SGLangEmbedding

# Default configuration
DB_PATH = "local/data/qdrant_hotpotqa"
COLLECTION_NAME = "hotpotqa_articles"

# Lazy-loaded globals
_embedding: SGLangEmbedding | None = None
_storage: QdrantStorage | None = None


def _get_components() -> tuple[SGLangEmbedding, QdrantStorage]:
    """Lazy initialization of embedding and storage."""
    global _embedding, _storage
    if _embedding is None:
        _embedding = SGLangEmbedding()
    if _storage is None:
        _storage = QdrantStorage(
            vector_dim=_embedding.get_output_dim(),
            path=DB_PATH,
            collection_name=COLLECTION_NAME,
        )
    return _embedding, _storage


def search_engine(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search Wikipedia articles for information related to the query.

    Use this tool to find relevant Wikipedia article snippets that may
    contain answers to factual questions.

    Args:
        query (str): The search query (e.g., "Scott Derrickson nationality").
        top_k (int): Number of results to return. (default: 5)

    Returns:
        List[Dict[str, Any]]: A list of search results, each containing:
            - 'result_id': Result number (1-indexed).
            - 'title': The Wikipedia article title.
            - 'description': A snippet of the article content.
            - 'url': A placeholder URL for the article.

            Example result:
            {
                'result_id': 1,
                'title': 'Scott Derrickson',
                'description': 'Scott Derrickson (born July 16, 1966) is an
                    American director, screenwriter and producer...',
                'url': 'https://en.wikipedia.org/wiki/Scott_Derrickson'
            }
    """
    embedding, storage = _get_components()

    query_vec = embedding.embed_list([query])[0]
    results = storage.query(VectorDBQuery(query_vector=query_vec, top_k=top_k))

    responses = []
    for i, r in enumerate(results, start=1):
        title = r.record.payload.get("title", "Unknown")
        text = r.record.payload.get("text", "")
        # Extract content after title
        content = text.split("\n", 1)[-1] if "\n" in text else text

        responses.append({
            "result_id": i,
            "title": title,
            "description": content[:500] + "..." if len(content) > 500 else content,
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        })

    return responses

