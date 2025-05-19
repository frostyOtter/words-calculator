import numpy as np
from typing import List, Tuple
from loguru import logger
from gensim.models.keyedvectors import KeyedVectors
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity


class GloveEmbeddings:
    def __init__(
        self,
        model_name: str = "glove-wiki-gigaword-50",
    ):
        """
        Initialize the GloVe embeddings model.

        Args:
            model_name: The name of the GloVe model to use (default: glove-wiki-gigaword-50)
        """
        self.model_name = model_name
        self.model = self._load_model()

    @lru_cache(maxsize=1)
    def _load_model(self) -> KeyedVectors:
        """
        Load the GloVe model using Gensim.
        The result is cached to avoid reloading the model multiple times.

        Returns:
            KeyedVectors: The loaded GloVe model
        """
        try:
            from gensim.downloader import load

            model = load(self.model_name)
            logger.info(f"Beep Boop, Loaded GloVe model: {self.model_name}")
            return model
        except ValueError as e:
            logger.error(f"Beep Boop, Loading model failed: {e}")
            raise

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        For GloVe, this will average the embeddings of all words in the text.

        Args:
            text: The text to generate embedding for

        Returns:
            numpy array containing the embedding
        """
        try:
            # Split text into words and get embeddings for each word
            words = text.lower().split()
            word_vectors = []

            for word in words:
                if word in self.model:
                    word_vectors.append(self.model[word])

            if not word_vectors:
                logger.warning(f"No known words found in text: {text}")
                return np.zeros(self.model.vector_size)

            # Average the word vectors
            return np.mean(word_vectors, axis=0)

        except Exception as e:
            logger.error(f"Beep Boop, Generating embedding failed: {e}")
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
            embeddings = [self.generate_embedding(text) for text in texts]
            return np.array(embeddings)

        except Exception as e:
            logger.error(f"Beep Boop, Generating embeddings failed: {e}")
            raise

    def find_closest_word(
        self, vector: np.ndarray, exclude_words: List[str]
    ) -> List[Tuple[str, float]]:
        """Finds the top 5 closest words in the vocabulary to the given vector."""
        similarities = cosine_similarity(vector.reshape(1, -1), self.model.vectors)[0]
        sorted_indices = np.argsort(similarities)[::-1]

        results: List[Tuple[str, float]] = []
        for index in sorted_indices:
            closest_word = self.model.index_to_key[index]
            if closest_word not in exclude_words:
                similarity = similarities[index]
                results.append((closest_word, similarity))
                if len(results) >= 5:
                    break
        return results


if __name__ == "__main__":
    # Example usage
    embeddings = GloveEmbeddings()
    single_embedding = embeddings.generate_embedding("Hello world")
    print(f"Single embedding shape: {single_embedding.shape}")

    multiple_embeddings = embeddings.generate_embeddings(
        ["Hello world", "Goodbye world"]
    )
    print(f"Multiple embeddings shape: {multiple_embeddings.shape}")
