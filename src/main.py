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
        ["GloVe", "Cohere", "OpenAI", "Azure OpenAI", "SBERT"],
        index=0,
        key="embedding_service",
    )

    try:
        if embedding_service == "GloVe":
            from glove_embeddings import GloveEmbeddings

            return GloveEmbeddings()
        elif embedding_service == "Cohere":
            from cohere_embeddings import CohereEmbeddings

            return CohereEmbeddings(api_key=os.getenv("COHERE_API_KEY"))
        elif embedding_service == "OpenAI":
            from openai_embeddings import OpenAIEmbeddings

            return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        elif embedding_service == "Azure OpenAI":
            from azure_openai_embeddings import AzureOpenAIEmbeddings

            return AzureOpenAIEmbeddings(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            )
        elif embedding_service == "SBERT":
            from sbert_embeddings import SBertEmbeddings

            return SBertEmbeddings()
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
    embedding_service: str,
    target_vector: np.ndarray,
    words: List[str],
) -> Optional[List[Tuple[str, float]]]:
    """Get similar words from LanceDB for API-based embedding services."""
    try:

        # Initialize LanceDB on first call
        db = EmbeddingsDB()
        db.connect()

        # Get table name based on embedding service
        table_name = f"{embedding_service.lower()}_embeddings"

        # Check if table exists
        if table_name not in db.db.table_names():
            st.error(
                f"Beep Boop, table '{table_name}' does not exist. Please ingest data first."
            )
            return None

        # Search in LanceDB
        search_results = db.search(table_name, target_vector, limit=5)

        # Format results
        similar_words = [
            (result["text"], float(result["_distance"]))
            for result in search_results
            if result not in words
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


def main() -> None:
    st.title("Word Analogy Calculator")

    st.subheader("Enter words with operations (- or +) between them")
    st.caption("Example: king - man + woman")
    st.caption("Press Enter when done.")

    embeddings_model = get_embedding_model()
    if embeddings_model is None:
        return

    with st.form("Enter words and operations:", enter_to_submit=True):
        analogy_input = st.text_input(label="", value="")
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

        embedding_service = st.session_state.get("embedding_service", "GloVe")

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
                embedding_service, target_word_embedding, words
            )
            if similar_words:
                display_similar_words(similar_words)
            else:
                st.warning("Could not find any suitable answers in the vocabulary.")


if __name__ == "__main__":
    main()
