import numpy as np
from typing import Protocol, List


class EmbeddingModel(Protocol):
    """Protocol defining the interface for embedding models.

    Think of this as a contract - like a blueprint that all embedding
    services must follow, similar to how all cars have steering wheels
    and brakes regardless of manufacturer.
    """

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for a single text."""
        ...

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embedding vectors for multiple texts."""
        ...
