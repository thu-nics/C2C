"""Simple FAISS searcher with SGLang embedding backend.

This provides the same search functionality as the MCP server but as a simple
callable function without MCP transport layers.
"""

import glob
import logging
import os
import pickle
from itertools import chain
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from rosetta.workflow.embeddings import SGLangEmbedding

logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_INDEX_PATH = "local/data/BrowseCompPlus/indexes/qwen3-embedding-8b/corpus.*.pkl"
DEFAULT_DATASET_NAME = "Tevatron/browsecomp-plus-corpus"
DEFAULT_SGLANG_URL = "http://localhost:30001"
DEFAULT_SGLANG_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_TASK_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
DEFAULT_SNIPPET_MAX_TOKENS = 512
DEFAULT_SNIPPET_TOKENIZER = "Qwen/Qwen3-Embedding-8B"

class BrowseCompPlusSearcher:
    """FAISS searcher using SGLang embedding backend."""

    def __init__(
        self,
        index_path: str,
        dataset_name: str = "Tevatron/browsecomp-plus-corpus",
        sglang_url: str = "http://localhost:30001",
        sglang_model: str = "Qwen/Qwen3-Embedding-8B",
        task_prefix: str = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        snippet_max_tokens: int | None = 512,
        snippet_tokenizer: str = "Qwen/Qwen3-0.6B",
    ):
        """
        Initialize the searcher.

        Args:
            index_path: Glob pattern for pickle files (e.g. /path/to/corpus.*.pkl)
            dataset_name: HuggingFace dataset name for document retrieval
            sglang_url: URL of SGLang server
            sglang_model: Model name for SGLang embeddings
            task_prefix: Prefix to add to queries before embedding
            snippet_max_tokens: Max tokens for snippets (set to None to disable truncation)
            snippet_tokenizer: Tokenizer to use for snippet truncation
        """
        self.task_prefix = task_prefix
        self.snippet_max_tokens = snippet_max_tokens

        # Initialize embedding model
        logger.info(f"Initializing SGLang embedding with {sglang_model}")
        self.embedding = SGLangEmbedding(url=sglang_url, model=sglang_model)

        # Load FAISS index
        logger.info(f"Loading FAISS index from {index_path}")
        self.index, self.lookup = self._load_faiss_index(index_path)

        # Load dataset for document retrieval
        logger.info(f"Loading dataset: {dataset_name}")
        self.docid_to_text = self._load_dataset(dataset_name)

        # Initialize snippet tokenizer if needed
        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            logger.info(f"Loading tokenizer for snippet truncation: {snippet_tokenizer}")
            self.tokenizer = AutoTokenizer.from_pretrained(snippet_tokenizer)

        logger.info("SimpleFaissSearcher initialized successfully")

    def _load_faiss_index(self, index_path: str) -> tuple[faiss.Index, list[str]]:
        """Load FAISS index from pickle files."""
        def pickle_load(path):
            with open(path, "rb") as f:
                reps, lookup = pickle.load(f)
            return np.array(reps), lookup

        index_files = glob.glob(index_path)
        logger.info(f"Pattern match found {len(index_files)} files")

        if not index_files:
            raise ValueError(f"No files found matching pattern: {index_path}")

        # Load first shard
        p_reps_0, p_lookup_0 = pickle_load(index_files[0])

        # Create FAISS index
        dimension = p_reps_0.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for similarity

        # Load all shards
        shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
        if len(index_files) > 1:
            shards = tqdm(shards, desc="Loading shards", total=len(index_files))

        lookup = []
        for p_reps, p_lookup in shards:
            index.add(p_reps)
            lookup += p_lookup

        # Try to move to GPU if available
        num_gpus = faiss.get_num_gpus()
        if num_gpus > 0:
            logger.info(f"Moving index to {num_gpus} GPU(s)")
            if num_gpus == 1:
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index, co)
            else:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.useFloat16 = True
                index = faiss.index_cpu_to_all_gpus(index, co, ngpu=num_gpus)
        else:
            logger.info("Using CPU for FAISS index")

        return index, lookup

    def _load_dataset(self, dataset_name: str) -> dict[str, str]:
        """Load dataset for document retrieval."""
        try:
            dataset_cache = os.getenv("HF_DATASETS_CACHE")
            cache_dir = dataset_cache if dataset_cache else None

            ds = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
            docid_to_text = {row["docid"]: row["text"] for row in ds}
            logger.info(f"Loaded {len(docid_to_text)} passages from dataset")
            return docid_to_text
        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_name}': {e}")
            raise

    def _truncate_snippet(self, text: str) -> str:
        """Truncate text to max tokens if configured."""
        if not self.snippet_max_tokens or not self.tokenizer:
            return text

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > self.snippet_max_tokens:
            truncated_tokens = tokens[:self.snippet_max_tokens]
            return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return text

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query string
            k: Number of results to return (default: 5)

        Returns:
            List of search results with docid, score, and snippet

            Example result:
            [
                {
                    'docid': 'doc123',
                    'score': 0.85,
                    'snippet': 'This is the document text...'
                },
                ...
            ]
        """
        # Encode query with task prefix
        query_with_prefix = self.task_prefix + query
        query_embedding = self.embedding.embed_list([query_with_prefix])[0]

        # Search FAISS index
        query_vec = np.array([query_embedding], dtype=np.float32)
        scores, indices = self.index.search(query_vec, k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            docid = self.lookup[idx]
            text = self.docid_to_text.get(docid, "Text not found")
            snippet = self._truncate_snippet(text)

            results.append({
                "docid": docid,
                "score": float(score),
                "snippet": snippet,
            })

        return results

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve full document by its ID.

        Args:
            docid: Document ID to retrieve

        Returns:
            Document dictionary with docid and text, or None if not found
        """
        text = self.docid_to_text.get(docid)
        if text is None:
            return None

        return {
            "docid": docid,
            "text": text,
        }

# Lazy-loaded global searcher
_searcher: BrowseCompPlusSearcher | None = None
_config: dict[str, Any] = {}

def configure_search(
    index_path: str = DEFAULT_INDEX_PATH,
    dataset_name: str = DEFAULT_DATASET_NAME,
    sglang_url: str = DEFAULT_SGLANG_URL,
    sglang_model: str = DEFAULT_SGLANG_MODEL,
    task_prefix: str = DEFAULT_TASK_PREFIX,
    snippet_max_tokens: int | None = DEFAULT_SNIPPET_MAX_TOKENS,
    snippet_tokenizer: str = DEFAULT_SNIPPET_TOKENIZER,
) -> None:
    """
    Configure the search engine settings.

    This function sets the configuration for the lazy-loaded searcher instance.
    Call this before using the search() function if you need non-default settings.

    Args:
        index_path: Glob pattern for FAISS pickle files (e.g., /path/to/corpus.*.pkl)
        dataset_name: HuggingFace dataset name for document retrieval
        sglang_url: URL of the SGLang embedding server
        sglang_model: Model name for SGLang embeddings
        task_prefix: Prefix to add to queries before embedding
        snippet_max_tokens: Max tokens for snippets (None to disable truncation)
        snippet_tokenizer: Tokenizer model for snippet truncation
    """
    global _config
    _config = {
        "index_path": index_path,
        "dataset_name": dataset_name,
        "sglang_url": sglang_url,
        "sglang_model": sglang_model,
        "task_prefix": task_prefix,
        "snippet_max_tokens": snippet_max_tokens,
        "snippet_tokenizer": snippet_tokenizer,
    }


def _get_searcher() -> BrowseCompPlusSearcher:
    """Lazy initialization of the searcher instance."""
    global _searcher
    if _searcher is None:
        # Use configured settings or defaults
        config = _config if _config else {
            "index_path": DEFAULT_INDEX_PATH,
            "dataset_name": DEFAULT_DATASET_NAME,
            "sglang_url": DEFAULT_SGLANG_URL,
            "sglang_model": DEFAULT_SGLANG_MODEL,
            "task_prefix": DEFAULT_TASK_PREFIX,
            "snippet_max_tokens": DEFAULT_SNIPPET_MAX_TOKENS,
            "snippet_tokenizer": DEFAULT_SNIPPET_TOKENIZER,
        }
        _searcher = BrowseCompPlusSearcher(**config)
    return _searcher


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search the BrowseComp-Plus corpus for relevant documents.

    Use this tool to find relevant document passages from the BrowseComp-Plus
    corpus (~100K human-verified documents) that may help answer reasoning-intensive
    queries.

    Args:
        query (str): The search query (e.g., "What is machine learning?").
        top_k (int): Number of results to return. (default: 5)

    Returns:
        List[Dict[str, Any]]: A list of search results, each containing:
            - 'docid': Document identifier.
            - 'score': Similarity score (higher is better).
            - 'snippet': Document text content (may be truncated based on configuration).

            Example result:
            [
                {
                    'docid': 'doc_12345',
                    'score': 0.8523,
                    'snippet': 'Machine learning is a subset of artificial
                        intelligence that enables systems to learn...'
                },
                {
                    'docid': 'doc_67890',
                    'score': 0.7891,
                    'snippet': 'Deep learning algorithms use neural networks...'
                }
            ]
    """
    searcher = _get_searcher()
    return searcher.search(query, k=top_k)


def get_document(docid: str) -> Optional[Dict[str, Any]]:
    """Retrieve the full text of a document by its ID.

    Use this tool to get the complete content of a specific document after
    finding it via search. This is useful when you need the full context
    beyond the truncated snippet.

    Args:
        docid (str): Document identifier from search results.

    Returns:
        Dict[str, Any] | None: Document dictionary with full text, or None if
            not found. Contains:
            - 'docid': Document identifier.
            - 'text': Full document text.

            Example result:
            {
                'docid': 'doc_12345',
                'text': 'Machine learning is a subset of artificial intelligence
                    that enables systems to learn and improve from experience...'
            }
    """
    searcher = _get_searcher()
    return searcher.get_document(docid)
