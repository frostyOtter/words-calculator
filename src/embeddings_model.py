import os
from loguru import logger
from functools import lru_cache
from typing import Optional, List, Callable
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(override=True)


@lru_cache(maxsize=128)
def glove_embedding(
    model_name: str = "glove-wiki-gigaword-50",
) -> Optional[KeyedVectors]:
    """Loads a pre-trained GloVe model using Gensim."""
    from gensim.downloader import load

    try:
        glove_model = load(model_name)
        return glove_model
    except ValueError as e:
        logger.error(f"Error loading GloVe model '{model_name}'.")
        return None


def cohere_embedding(
    model_name: str = "embed-english-v3.0",
) -> Callable[[List[str]], np.ndarray]:
    import cohere as co

    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

    try:
        client = co.ClientV2(api_key=COHERE_API_KEY)  # type: ignore
        ## Cohere's API has a limit of 96 requests per minute, so we need to batch the requests
        BATCH_SIZE = 96

        def embed(texts: List[str], cohere_model_name: str = model_name) -> np.ndarray:
            all_embeddings = []

            try:
                for i in tqdm(
                    range(0, len(texts), BATCH_SIZE), desc="Generating embeddings"
                ):
                    batch = texts[i : i + BATCH_SIZE]

                    response = client.embed(
                        texts=batch,
                        model=cohere_model_name,
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

                logger.debug(f"Successfully embedded {len(texts)} texts")
                return np.array(all_embeddings)

            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                raise

        return embed

    except ValueError as e:
        logger.error(f"Error loading Cohere model: {e}")
        return None


def openai_embedding(
    model_name: str = "text-embedding-3-small",
) -> Callable[[List[str]], np.ndarray]:
    import openai

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        def embed(texts: List[str]) -> np.ndarray:
            try:
                return client.embed(texts=texts, model=model_name).embeddings
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                raise

        return embed
    except Exception as e:
        logger.error(f"Error loading OpenAI model: {e}")
        return None


def azure_openai_embedding(
    model_name: str = "text-embedding-3-large",
) -> Callable[[List[str]], np.ndarray]:
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    from openai import AzureOpenAI

    try:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-12-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )

        def embed(texts: List[str]) -> np.ndarray:
            try:
                response = client.embeddings.create(input=texts, model=model_name)
                return np.array([embedding.embedding for embedding in response.data])
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                raise

        return embed
    except Exception as e:
        logger.error(f"Error loading Azure OpenAI model: {e}")
        return None


def sbert_embedding(
    model_name: str = "all-MiniLM-L6-v2",
) -> Callable[[List[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(model_name)

        def embed(texts: List[str]) -> np.ndarray:
            try:
                return model.encode(texts)
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                raise

        return embed
    except Exception as e:
        logger.error(f"Error loading SBERT model: {e}")
        return None


if __name__ == "__main__":
    model = glove_embedding()
    logger.info(model.vectors.shape)
