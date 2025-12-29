"""SGLang embedding adapter for CAMEL."""

from typing import Any

import requests
from camel.embeddings import BaseEmbedding


class SGLangEmbedding(BaseEmbedding):
    """Embedding model using SGLang's OpenAI-compatible API."""

    def __init__(
        self,
        url: str = "http://localhost:30001",
        model: str = "Qwen/Qwen3-Embedding-8B",
    ):
        self.url = url
        self.model = model
        self._dim: int | None = None

    def embed_list(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Embed a list of texts."""
        resp = requests.post(
            f"{self.url}/v1/embeddings",
            json={"input": texts, "model": self.model},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]

    def get_output_dim(self) -> int:
        """Get embedding dimension (lazy detection)."""
        if self._dim is None:
            self._dim = len(self.embed_list(["test"])[0])
        return self._dim

