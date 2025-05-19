import os
from typing import List, Optional
import numpy as np
from loguru import logger
from tqdm import tqdm
import cohere


class CohereEmbeddings:
    def __init__(
        self,
        api_key: str,
        model_name: str = "embed-english-v3.0",
        batch_size: int = 96,
    ):
        """
        Initialize the Cohere embeddings client.

        Args:
            model_name: The name of the Cohere model to use
            api_key: Optional API key. If not provided, will use COHERE_API_KEY from environment
            batch_size: Size of batches for embedding generation (default: 96, Cohere's rate limit)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.api_key = api_key

        try:
            self.client = cohere.ClientV2(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Beep Boop, Initializing failed: {e}")
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
            response = self.client.embed(
                texts=[text],
                model=self.model_name,
                input_type="search_query",
                embedding_types=["float"],
            )

            if hasattr(response.embeddings, "float_"):
                return np.array(response.embeddings.float_[0])
            elif hasattr(response.embeddings, "float"):
                return np.array(response.embeddings.float[0])
            else:
                raise ValueError(
                    "Cannot find 'float_' or 'float' in response.embeddings"
                )

        except Exception as e:
            logger.error(f"Beep Boop, Generating embedding failed: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts in batches.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            numpy array containing all embeddings
        """
        all_embeddings = []

        try:
            for i in tqdm(
                range(0, len(texts), self.batch_size), desc="Generating embeddings"
            ):
                batch = texts[i : i + self.batch_size]

                response = self.client.embed(
                    texts=batch,
                    model=self.model_name,
                    input_type="search_query",
                    embedding_types=["float"],
                )

                if hasattr(response.embeddings, "float_"):
                    all_embeddings.extend(response.embeddings.float_)
                elif hasattr(response.embeddings, "float"):
                    all_embeddings.extend(response.embeddings.float)
                else:
                    raise ValueError(
                        "Cannot find 'float_' or 'float' in response.embeddings"
                    )

            return np.array(all_embeddings)

        except Exception as e:
            logger.error(f"Beep Boop, Generating embeddings failed: {e}")
            raise


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    embeddings = CohereEmbeddings(api_key=os.getenv("COHERE_API_KEY"))
    single_embedding = embeddings.generate_embedding("Hello, world!")
    print(f"Single embedding shape: {single_embedding.shape}")
