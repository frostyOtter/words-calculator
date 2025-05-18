# import sys
# import pickle
# import os
# import asyncio
# from tqdm import tqdm
# from tqdm.asyncio import tqdm_asyncio
# import numpy as np
# from embeddings_model import (
#     glove_embedding,
#     cohere_embedding,
#     openai_embedding,
#     azure_openai_embedding,
#     sbert_embedding,
# )
# from loguru import logger


# def load_words(file_path: str) -> list[str]:
#     """Load words from the specified file."""
#     with open(file_path, "r") as f:
#         return [line.strip() for line in f if line.strip()]


# def get_embedding_function(service_name: str):
#     """Get the appropriate embedding function based on service name."""
#     service_map = {
#         "glove": glove_embedding,
#         "cohere": cohere_embedding,
#         "openai": openai_embedding,
#         "azure_openai": azure_openai_embedding,
#         "sbert": sbert_embedding,
#     }

#     if service_name not in service_map:
#         raise ValueError(
#             f"Unknown service: {service_name}. Available services: {list(service_map.keys())}"
#         )

#     return service_map[service_name]()


# async def process_batch_async(
#     words: list[str], embed_fn, batch_size: int = 100, max_concurrent: int = 50
# ) -> list:
#     """Process words in batches concurrently and return combined embeddings.

#     Args:
#         words: List of words to process
#         embed_fn: Embedding function to use
#         batch_size: Number of words to process in each batch
#         max_concurrent: Maximum number of concurrent requests

#     Returns:
#         List of embeddings for all words
#     """
#     # Create batches
#     batches = [words[i : i + batch_size] for i in range(0, len(words), batch_size)]
#     total_batches = len(batches)

#     # Create semaphore to limit concurrent requests
#     semaphore = asyncio.Semaphore(max_concurrent)

#     async def process_with_semaphore(batch):
#         async with semaphore:
#             return await embed_fn(batch)

#     # Process batches with progress bar
#     all_embeddings = []
#     tasks = [process_with_semaphore(batch) for batch in batches]

#     for batch_embeddings in await tqdm_asyncio.gather(
#         *tasks, desc="Processing batches"
#     ):
#         all_embeddings.extend(batch_embeddings)

#     return all_embeddings


# async def main_async(service_name: str):
#     """Generate and save embeddings for the specified service.

#     Args:
#         service_name: Name of the embedding service to use (glove, cohere, openai, azure_openai, sbert)
#     """
#     # Create embeddings directory if it doesn't exist
#     embeddings_dir = os.path.join("data", "embeddings")
#     os.makedirs(embeddings_dir, exist_ok=True)

#     # Load words
#     words = load_words(os.path.join("data", "Words_Alpha.txt"))
#     logger.info(f"Loaded {len(words)} words")

#     # Get embedding function
#     embed_fn = get_embedding_function(service_name)

#     # For Azure OpenAI, use async mode
#     if service_name == "azure_openai":
#         embed_fn = azure_openai_embedding(async_mode=True)

#     # Generate embeddings in batches
#     logger.info(f"Generating embeddings using {service_name} in batches of 100...")
#     embeddings = await process_batch_async(words, embed_fn)

#     # Save embeddings
#     output_path = os.path.join(embeddings_dir, f"{service_name}_embeddings.pkl")
#     with open(output_path, "wb") as f:
#         pickle.dump({"words": words, "embeddings": embeddings}, f)

#     logger.info(f"Saved embeddings to {output_path}")


# def process_batch(words: list[str], embed_fn, batch_size: int = 100) -> list:
#     """Process words in batches and return combined embeddings.

#     Args:
#         words: List of words to process
#         embed_fn: Embedding function to use
#         batch_size: Number of words to process in each batch

#     Returns:
#         List of embeddings for all words
#     """
#     all_embeddings = []
#     total_batches = (len(words) + batch_size - 1) // batch_size

#     for i in tqdm(
#         range(0, len(words), batch_size), total=total_batches, desc="Processing batches"
#     ):
#         batch = words[i : i + batch_size]
#         batch_embeddings = embed_fn(batch)
#         all_embeddings.extend(batch_embeddings)

#     return all_embeddings


# def main(service_name: str):
#     """Generate and save embeddings for the specified service.

#     Args:
#         service_name: Name of the embedding service to use (glove, cohere, openai, azure_openai, sbert)
#     """
#     # Create embeddings directory if it doesn't exist
#     embeddings_dir = os.path.join("data", "embeddings")
#     os.makedirs(embeddings_dir, exist_ok=True)

#     # Load words
#     words = load_words(os.path.join("data", "Words_Alpha.txt"))
#     logger.info(f"Loaded {len(words)} words")

#     # Get embedding function
#     embed_fn = get_embedding_function(service_name)

#     # Generate embeddings in batches
#     logger.info(f"Generating embeddings using {service_name} in batches of 100...")
#     embeddings = process_batch(words, embed_fn)

#     # Save embeddings
#     output_path = os.path.join(embeddings_dir, f"{service_name}_embeddings.pkl")
#     with open(output_path, "wb") as f:
#         pickle.dump({"words": words, "embeddings": embeddings}, f)

#     logger.info(f"Saved embeddings to {output_path}")


# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         logger.error("Usage: python pre_compute_embeddings.py <service_name>")
#         sys.exit(1)

#     service_name = sys.argv[1]
#     if service_name == "azure_openai":
#         asyncio.run(main_async(service_name))
#     else:
#         main(service_name)
