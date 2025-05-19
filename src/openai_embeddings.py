import os
from typing import List
import numpy as np
from loguru import logger
from tqdm import tqdm
from openai import OpenAI


class OpenAIEmbeddings:
    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 16,
    ):
        """
        Initialize the OpenAI embeddings client.

        Args:
            api_key: OpenAI API key
            model_name: The name of the OpenAI model to use
            batch_size: Size of batches for embedding generation (default: 16)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.api_key = api_key

        try:
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
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
            response = self.client.embeddings.create(
                input=[text], model=self.model_name
            )
            return np.array(response.data[0].embedding)

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
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
                response = self.client.embeddings.create(
                    input=batch, model=self.model_name
                )
                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)

            logger.debug(f"Successfully embedded {len(texts)} texts")
            return np.array(all_embeddings)

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    single_embedding = embeddings.generate_embedding("Hello, world!")
    print(f"Single embedding shape: {single_embedding.shape}")
