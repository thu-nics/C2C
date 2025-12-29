"""Local search engine retriever for HotpotQA articles."""

from typing import Any

from camel.storages import FaissStorage, VectorDBQuery

from rosetta.workflow.embeddings import SGLangEmbedding

# Default configuration
DB_PATH = "local/data/faiss_hotpotqa"
COLLECTION_NAME = "hotpotqa_articles"

# Lazy-loaded globals
_embedding: SGLangEmbedding | None = None
_storage: FaissStorage | None = None


def _get_components() -> tuple[SGLangEmbedding, FaissStorage]:
    """Lazy initialization of embedding and storage."""
    global _embedding, _storage
    if _embedding is None:
        _embedding = SGLangEmbedding()
    if _storage is None:
        _storage = FaissStorage(
            vector_dim=_embedding.get_output_dim(),
            storage_path=DB_PATH,
            collection_name=COLLECTION_NAME,
            index_type="IVFFlat",  # Faster search for large datasets
        )
    return _embedding, _storage


def search_engine(query: str, top_k: int = 5, reverse: bool = False) -> list[dict[str, Any]]:
    """Search local wiki database for information related to the query.

    Use this tool to find relevant local wiki database article snippets that may
    contain answers to factual questions.

    Args:
        query (str): The search query (e.g., "Scott Derrickson nationality").
        top_k (int): Number of results to return. (default: 5)
        reverse (bool): Whether to reverse the order of results. (default: True)

    Returns:
        List[Dict[str, Any]]: A list of search results, each containing:
            - 'result_id': Result number (1-indexed).
            - 'title': The local wiki database article title.
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

    ordered_results = reversed(results) if reverse else results

    responses = []
    for i, r in enumerate(ordered_results, start=1):
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
