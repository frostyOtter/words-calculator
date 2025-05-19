import numpy as np
from typing import List
from loguru import logger
from functools import lru_cache


class SBertEmbeddings:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the SBERT embeddings model.

        Args:
            model_name: The name of the SBERT model to use (default: all-MiniLM-L6-v2)
        """
        self.model_name = model_name
        self.model = self._load_model()

    @lru_cache(maxsize=1)
    def _load_model(self):
        """
        Load the SBERT model.
        The result is cached to avoid reloading the model multiple times.

        Returns:
            SentenceTransformer: The loaded SBERT model
        """
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name)
            logger.info(f"Successfully loaded SBERT model: {self.model_name}")
            return model
        except Exception as e:
            logger.error(f"Error loading SBERT model '{self.model_name}': {e}")
            raise

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: The text to generate embedding for

        Returns:
            numpy array containing the embedding
        """
        try:
            # SBERT can handle the text directly
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            numpy array containing all embeddings
        """
        try:
            # SBERT can handle multiple texts efficiently
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.debug(f"Successfully embedded {len(texts)} texts")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    embeddings = SBertEmbeddings()
    single_embedding = embeddings.generate_embedding("Hello world")
    print(f"Single embedding shape: {single_embedding.shape}")

    multiple_embeddings = embeddings.generate_embeddings(
        ["Hello world", "Goodbye world"]
    )
    print(f"Multiple embeddings shape: {multiple_embeddings.shape}")
