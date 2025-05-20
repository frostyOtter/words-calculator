import os
import streamlit as st
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

from typing import Optional, Tuple, List, Callable
import yaml
from models import SimilarWords
from loguru import logger
from embeddings_db import EmbeddingsDB


def preprocess_analogy_string(
    analogy_string: str,
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Preprocesses the analogy string and returns words and operations if valid."""
    # Clean and normalize the input string
    analogy_string = analogy_string.lower().strip()

    # Split the string into words and operations
    parts = analogy_string.split()
    if len(parts) < 2 or len(parts) > 10:
        st.warning(
            "Please enter between 2 and 10 words with operations (- or +) between them."
        )
        return None, None

    # Check if the input has valid operations
    if not all(part in ["-", "+"] for part in parts[1::2]):
        st.warning("Please use only '-' and '+' as operations between words.")
        return None, None

    # Get words and operations
    words = parts[::2]  # Get all words (even indices)
    operations = parts[1::2]  # Get all operations (odd indices)

    return words, operations


def perform_word_analogy(
    embeddings_model: Callable, words: List[str], operations: List[str]
) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """Performs the word analogy calculation with support for multiple operations."""

    try:
        # Start with the first word's vector
        words_embeddings = embeddings_model.generate_embeddings(words)
        target_word_embedding = words_embeddings[0]
        # Apply operations in sequence
        for i, op in enumerate(operations):
            if op == "-":
                target_word_embedding = target_word_embedding - words_embeddings[i + 1]
            else:  # op == '+'
                target_word_embedding = target_word_embedding + words_embeddings[i + 1]

        return target_word_embedding, words_embeddings
    except Exception as e:
        st.warning(f"Error performing word analogy: {e}")
        return None, None


def get_embedding_model() -> Callable:
    """Returns the selected embedding model based on user choice."""
    from dotenv import load_dotenv

    load_dotenv(override=True)
    import os

    embedding_service = st.selectbox(
        "Select Embedding Service",
        ["GloVe", "Cohere", "OpenAI", "Azure_OpenAI", "SBERT"],
        index=1,
        key="embedding_service",
    )

    try:
        if embedding_service == "GloVe":
            from glove_embeddings import GloveEmbeddings

            return GloveEmbeddings()
        elif embedding_service == "Cohere":
            from cohere_embeddings import CohereEmbeddings

            api_key = os.getenv("COHERE_API_KEY")
            if api_key is None:
                st.error("Beep Boop, Cohere API key is not set.")
                return None
            return CohereEmbeddings(api_key=api_key)
        elif embedding_service == "OpenAI":
            from openai_embeddings import OpenAIEmbeddings

            model_name = os.getenv("OPENAI_MODEL")
            api_key = os.getenv("OPENAI_API_KEY")
            if any(model_name, api_key) is None:
                st.error("Beep Boop, OpenAI model or API key is not set.")
                return None
            return OpenAIEmbeddings(
                api_key=os.getenv("OPENAI_API_KEY"), model_name=model_name
            )
        elif embedding_service == "Azure_OpenAI":
            from azure_openai_embeddings import AzureOpenAIEmbeddings

            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            if any(api_key, endpoint, model_name) is None:
                st.error("Beep Boop, Azure OpenAI model or API key is not set.")
                return None
            return AzureOpenAIEmbeddings(
                api_key=api_key,
                endpoint=endpoint,
                model_name=model_name,
            )
        elif embedding_service == "SBERT":
            from sbert_embeddings import SBertEmbeddings

            model_name = st.text_input(
                label="Enter model name", value="all-MiniLM-L6-v2"
            )
            if model_name is None:
                st.error("Beep Boop, model name is not set.")
                return None
            return SBertEmbeddings(model_name=model_name)
    except Exception as e:
        st.error(f"Error loading {embedding_service} model: {str(e)}")
        return None


def get_llm_model() -> Callable:
    """Returns the selected LLM model based on user choice."""
    from gemini_api_service import GeminiServiceProvider
    from dotenv import load_dotenv

    load_dotenv()
    import os

    return GeminiServiceProvider(api_key=os.getenv("GEMINI_API_KEY"))


def get_similar_words_from_db(
    embeddings_db: EmbeddingsDB,
    embedding_service: str,
    target_vector: np.ndarray,
    words: List[str],
) -> Optional[List[Tuple[str, float]]]:
    """Get similar words from LanceDB for API-based embedding services."""
    try:

        # Get table name based on embedding service
        table_name = f"{embedding_service.lower()}_embeddings"

        # Check if table exists
        if table_name not in embeddings_db.db.table_names():
            st.error(
                f"Beep Boop, table '{table_name}' does not exist. Please ingest data first."
            )
            return None

        # Search in LanceDB
        search_results = embeddings_db.search(table_name, target_vector, limit=5)

        # Format results
        similar_words = [
            (result["text"], float(result["_distance"]))
            for result in search_results
            if result["text"] not in words
        ]
        return similar_words

    except Exception as e:
        logger.error(f"Error getting similar words from DB: {e}")
        st.error(f"Error getting similar words: {str(e)}")
        return None


def get_llm_similar_words(
    llm_model: Callable, analogy_input: str, embeddings_model
) -> List[Tuple[str, float]]:
    pass


def display_similar_words(similar_words: List[Tuple[str, float]]) -> None:
    """Displays the similar words in a table."""
    st.write("Similar words:")
    # Sort similar words by similarity
    similar_words.sort(key=lambda x: x[1], reverse=True)
    for word, similarity in similar_words:
        st.write(f"- {word} (Similarity: {similarity:.2f})")


def add_new_data_to_db(
    table_row_count: int,
    embeddings_db: EmbeddingsDB,
    embeddings_model: Callable,
    embeddings_service: str,
    num_words: int,
) -> bool:

    with open(os.path.join("data", "20k-english-words.txt"), "r") as f:
        data = f.read().splitlines()
    st.write("Ingesting new words into database...")
    if len(data) == table_row_count:
        st.warning("Beep Boop, Database already contains all words.")
        return
    elif len(data) > table_row_count + num_words:
        data = data[table_row_count : table_row_count + num_words]
    else:
        data = data[table_row_count : len(data)]

    result = embeddings_db.add_new_data_to_table(
        embeddings_service.lower() + "_embeddings", embeddings_model, data
    )
    if result:
        logger.debug(f"Beep Boop, ingestion successful.")
    return result


def main() -> None:
    st.title("Word Analogy Calculator")

    st.subheader("Enter words with operations (- or +) between them")
    st.caption("Example: king - man + woman")
    st.caption("Press Enter when done.")

    embeddings_model = get_embedding_model()
    if embeddings_model is None:
        return
    embedding_service = st.session_state.get("embedding_service", "GloVe")
    if embedding_service != "GloVe":
        # Initialize LanceDB on first call
        embeddings_db = EmbeddingsDB()
        embeddings_db.connect()
        if (
            embedding_service.lower() + "_embeddings"
            not in embeddings_db.db.table_names()
        ):
            st.warning(
                f"Beep Boop, table '{embedding_service.lower()}_embeddings' does not exist. Please ingest data first."
            )
            table_row_count = 0
        else:
            table_row_count = embeddings_db.retrieve_table_row_count(
                embedding_service.lower() + "_embeddings"
            )
            if table_row_count == 0:
                st.warning(
                    "Beep Boop, No words found in the database. Please ingest data first."
                )
            else:
                st.write(f"Beep Boop, Found {table_row_count} words in the database.")
        add_data_result = False
        with st.expander("Ingest more words into database"):
            # Display a form, user can input number of words to embeddings into database.
            with st.form("Ingest words into database:"):
                num_words = st.number_input(
                    label="Number of words to ingest", min_value=1, max_value=1000
                )
                submit_button = st.form_submit_button("Submit")
                if submit_button:
                    st.write(
                        f"Beep Boop, Ingesting {num_words} words into the database, please wait...."
                    )
                    add_data_result = add_new_data_to_db(
                        table_row_count,
                        embeddings_db,
                        embeddings_model,
                        embedding_service,
                        num_words,
                    )
        if add_data_result:
            st.success("Beep Boop, Ingestion successful.")
            embeddings_db = None
            st.rerun()

    with st.form("Input words and operations", enter_to_submit=True):
        analogy_input = st.text_input(label="Input words and operations", value="")
        submit_button = st.form_submit_button("Submit")

    if submit_button and analogy_input.strip():
        # Preprocess user input message into words and operations
        words, operations = preprocess_analogy_string(analogy_input)
        if words is None or operations is None:
            return

        # Perform word analogy calculation
        target_word_embedding, _ = perform_word_analogy(
            embeddings_model, words, operations
        )

        if embedding_service == "GloVe":
            similar_words = embeddings_model.find_closest_word(
                target_word_embedding, words
            )
            if similar_words:
                display_similar_words(similar_words)
            else:
                st.warning("Beep Boop, GloVe can't find any similar words.")
        elif embedding_service == "SBERT":
            st.error("Beep Boop, SBERT embeddings are not implemented yet.")
        else:  # Cohere, OpenAI, or Azure OpenAI
            similar_words = get_similar_words_from_db(
                embeddings_db, embedding_service, target_word_embedding, words
            )
            if similar_words:
                display_similar_words(similar_words)
            else:
                st.warning("Could not find any suitable answers in the vocabulary.")


if __name__ == "__main__":
    main()
