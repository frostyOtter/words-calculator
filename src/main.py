import os
import streamlit as st
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Tuple, List, Callable
import yaml
from models import SimilarWords
from loguru import logger


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
    model: KeyedVectors, analogy_string: str
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Performs the word analogy calculation with support for multiple operations."""
    # Preprocess the analogy string
    words, operations = preprocess_analogy_string(analogy_string)
    missing_words = [word for word in words if word not in model]
    # Check if all words are in the vocabulary

    if missing_words:
        st.warning(
            f"The following word(s) are not in the vocabulary: {', '.join(missing_words)}"
        )
        return None, None

    try:
        # Start with the first word's vector
        result_vector = model[words[0]]

        # Apply operations in sequence
        for i, op in enumerate(operations):
            if op == "-":
                result_vector = result_vector - model[words[i + 1]]
            else:  # op == '+'
                result_vector = result_vector + model[words[i + 1]]

        return result_vector, words
    except KeyError as e:
        st.warning(f"Error accessing word vector: {e}")
        return None, None


def find_closest_word(
    model: KeyedVectors, vector: np.ndarray, exclude_words: List[str]
) -> List[Tuple[str, float]]:
    """Finds the top 5 closest words in the vocabulary to the given vector."""
    if vector is None:
        return []

    similarities = cosine_similarity(vector.reshape(1, -1), model.vectors)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    results: List[Tuple[str, float]] = []
    for index in sorted_indices:
        closest_word = model.index_to_key[index]
        if closest_word not in exclude_words:
            similarity = similarities[index]
            results.append((closest_word, similarity))
            if len(results) >= 5:
                break
    return results


def get_embedding_model() -> Callable:
    """Returns the selected embedding model based on user choice."""
    from embeddings_model import (
        glove_embedding,
        cohere_embedding,
        openai_embedding,
        azure_openai_embedding,
        sbert_embedding,
    )

    embedding_service = st.selectbox(
        "Select Embedding Service",
        ["GloVe", "Cohere", "OpenAI", "Azure OpenAI", "SBERT"],
        index=0,
        key="embedding_service",
    )

    try:
        if embedding_service == "GloVe":
            return glove_embedding()
        elif embedding_service == "Cohere":
            return cohere_embedding()
        elif embedding_service == "OpenAI":
            return openai_embedding()
        elif embedding_service == "Azure OpenAI":
            return azure_openai_embedding()
        elif embedding_service == "SBERT":
            return sbert_embedding()
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


def get_glove_similar_words(
    embeddings_model: KeyedVectors, analogy_input: str
) -> List[Tuple[str, float]]:
    """Calculates and displays similar words based on the analogy input."""
    result_vector, input_words = perform_word_analogy(embeddings_model, analogy_input)
    if result_vector is not None:
        similar_words = find_closest_word(embeddings_model, result_vector, input_words)
    else:
        return None
    return similar_words


def get_embeddings_from_api_service(
    words_input: List[str], similar_words: List[str], embeddings_model: Callable
) -> np.ndarray:
    # Embeddings analogy input and similar words
    embeddings = embeddings_model(words_input + similar_words)
    # Slide the embeddings results that match with the len of words and similar_words
    words_embeddings = embeddings[: len(words_input)]
    similar_words_embeddings = embeddings[len(words_input) :]
    if words_embeddings is None or similar_words_embeddings is None:
        return None
    return words_embeddings, similar_words_embeddings


def get_llm_similar_words(
    llm_model: Callable, analogy_input: str, embeddings_model
) -> List[Tuple[str, float]]:
    """Calculates and displays similar words based on the analogy input."""
    with open(os.path.join("src", "prompts.yaml"), "r") as f:
        prompts = yaml.safe_load(f)
    system_prompt = prompts["system_prompt"]
    user_prompt = prompts["user_prompt"]
    assistant_prompt = prompts["assistant_prompt"]
    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "ai", "content": assistant_prompt},
        {"role": "user", "content": analogy_input},
    ]

    similar_words = llm_model.generate_response(prompts, SimilarWords)
    similar_words = similar_words.get("similar_words")
    logger.info(f"Similar words: {similar_words}")
    words, operations = preprocess_analogy_string(analogy_input)
    words_embeddings, similar_words_embeddings = get_embeddings_from_api_service(
        words, similar_words, embeddings_model
    )

    if words_embeddings is None or similar_words_embeddings is None:
        return None

    # We are now have words, similar words from LLM, and embeddings
    # Calculate the operations between the words and similar words
    target_word_embeddings = words_embeddings[0]
    for i, op in enumerate(operations):
        if op == "-":
            target_word_embeddings = target_word_embeddings - words_embeddings[i + 1]
        else:
            target_word_embeddings = target_word_embeddings + words_embeddings[i + 1]

    similarities = [
        cosine_similarity(
            target_word_embeddings.reshape(1, -1),
            similar_word_embeddings.reshape(1, -1),
        )
        for similar_word_embeddings in similar_words_embeddings
    ]
    # Return List of tuples (word, similarity), also change from numpy.ndarray to float
    similar_words = [
        (word, float(similarity.item()))
        for word, similarity in zip(similar_words, similarities)
    ]
    similar_words = sorted(similar_words, key=lambda x: x[1], reverse=True)
    return similar_words


def display_similar_words(similar_words: List[Tuple[str, float]]) -> None:
    """Displays the similar words in a table."""
    st.write("Similar words:")
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

    llm_model = get_llm_model()
    if llm_model is None:
        return

    analogy_input = st.text_input(
        "Enter words and operations:", "", key="analogy_input"
    )

    if analogy_input:
        # Get the selected embedding service name
        embedding_service = st.session_state.get("embedding_service", "GloVe")

        if embedding_service == "GloVe":
            similar_words = get_glove_similar_words(embeddings_model, analogy_input)
            if similar_words:
                display_similar_words(similar_words)
            else:
                st.warning("Could not find any suitable answers in the vocabulary.")
        else:
            similar_words = get_llm_similar_words(
                llm_model, analogy_input, embeddings_model
            )
            if similar_words:
                display_similar_words(similar_words)
            else:
                st.warning("Could not find any suitable answers in the vocabulary.")


if __name__ == "__main__":
    main()
