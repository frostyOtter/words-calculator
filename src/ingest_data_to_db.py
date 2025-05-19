import os
import random
from loguru import logger
from dotenv import load_dotenv

if __name__ == "__main__":

    import sys

    data_path = sys.argv[1]

    from embeddings_db import EmbeddingsDB
    from cohere_embeddings import CohereEmbeddings

    db = EmbeddingsDB()
    db.connect()

    with open(data_path, "r") as f:
        data = f.read().splitlines()  # Split the input into a list of strings
    # random sample 10 batches of cohere embeddings
    data = random.sample(data, 960)

    load_dotenv(override=True)
    cohere_embeddings = CohereEmbeddings(api_key=os.getenv("COHERE_API_KEY"))
    embeddings_data = cohere_embeddings.generate_embeddings(data)

    data = [
        {"text": text, "vector": vector} for text, vector in zip(data, embeddings_data)
    ]

    db.create_table("cohere_embeddings", data)

    logger.info("Beep Boop, Data ingested successfully")
