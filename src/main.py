import os
import streamlit as st
import numpy as np
from typing import Optional, Tuple, List, Callable, Union, Any
from loguru import logger
from lance_embeddings_db import EmbeddingsDB


def preprocess_analogy_string(
    analogy_string: str,
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Preprocesses the analogy string and returns words and operations if valid."""
    # Clean and normalize the input string
    analogy_string = analogy_string.lower().strip()

    # Split the string into words and operations
    parts: list[str] = analogy_string.split()
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
    words: list[str] = parts[::2]  # Get all words (even indices)
    operations: list[str] = parts[1::2]  # Get all operations (odd indices)

    return words, operations


def perform_word_analogy(
    embeddings_model: Any, words: List[str], operations: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs the word analogy calculation with support for multiple operations."""
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


def get_embedding_model(
    embedding_service: str, service_configs: dict
) -> Union[Any | None]:
    """Returns the selected embedding model based on user choice.
    embedding_service: str - The embedding service to use.
    Supports: GloVe, Cohere, OpenAI, Azure_OpenAI, SBERT
    """
    try:
        if embedding_service == "GloVe":
            from glove_embeddings import GloveEmbeddings

            return GloveEmbeddings()
        elif embedding_service == "Cohere":
            from cohere_embeddings import CohereEmbeddings

            return CohereEmbeddings(
                api_key=service_configs["api_key"],
                model_name=service_configs["model_name"],
            )

        elif embedding_service == "OpenAI":
            from openai_embeddings import OpenAIEmbeddings

            return OpenAIEmbeddings(
                api_key=service_configs["api_key"],
                model_name=service_configs["model_name"],
            )

        elif embedding_service == "Azure_OpenAI":
            from azure_openai_embeddings import AzureOpenAIEmbeddings

            return AzureOpenAIEmbeddings(
                api_key=service_configs["api_key"],
                endpoint=service_configs["endpoint"],
                model_name=service_configs["model_name"],
            )

        elif embedding_service == "SBERT":
            from sbert_embeddings import SBertEmbeddings

            return SBertEmbeddings(model_name=service_configs["model_name"])

    except Exception as e:
        st.error(f"Error loading {embedding_service} model: {str(e)}")
        return None


def get_llm_model() -> Any:
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

        if embeddings_db.db is None:
            logger.error("Database object is not properly initialized")
            return None
        # Get table name based on embedding service
        table_name: str = f"{embedding_service.lower()}_embeddings"
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
) -> None:
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
    embeddings_model: Any,
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


def on_change_embedding_service():
    st.session_state["service_configs"] = None
    st.session_state["embedding_service"] = None


def main() -> None:
    st.title("Word Analogy Calculator")

    ## User select service and input service configs
    with st.container():
        embedding_service = st.selectbox(
            "Select embedding service",
            ["GloVe", "Cohere", "OpenAI", "Azure_OpenAI", "SBERT"],
            index=1,
            on_change=on_change_embedding_service,
        )

        if embedding_service == "GloVe":
            service_configs = {
                "model_name": st.text_input(
                    "Enter GloVe model name", "glove-wiki-gigaword-50"
                )
            }
        elif embedding_service == "Cohere":
            service_configs = {
                "api_key": st.text_input("Enter Cohere API key", type="password"),
                "model_name": st.text_input(
                    "Enter Cohere model name", "embed-english-v3.0"
                ),
            }
        elif embedding_service == "OpenAI":
            service_configs = {
                "api_key": st.text_input("Enter OpenAI API key", type="password"),
                "model_name": st.text_input(
                    "Enter OpenAI model name", "text-embedding-3-large"
                ),
            }
        elif embedding_service == "Azure_OpenAI":
            service_configs = {
                "api_key": st.text_input("Enter Azure OpenAI API key", type="password"),
                "model_name": st.text_input(
                    "Enter Azure OpenAI model name", "text-embedding-3-large"
                ),
                "endpoint": st.text_input("Enter Azure OpenAI endpoint"),
            }
        elif embedding_service == "SBERT":
            service_configs = {
                "model_name": st.text_input(
                    "Enter SBERT model name", "all-MiniLM-L6-v2"
                )
            }

        submit_button = st.button("Submit")
        if submit_button:
            st.session_state["service_configs"] = service_configs
            st.session_state["embedding_service"] = embedding_service
            st.rerun()

    ## After submit button is clicked, load the service configs and embedding service
    service_configs = st.session_state.get("service_configs")
    embedding_service = st.session_state.get("embedding_service")
    if service_configs and embedding_service:
        embeddings_model: Any | None = get_embedding_model(
            embedding_service, service_configs
        )
        if embeddings_model:

            ## Checking if there are any data for the selected embedding service in the database
            if embedding_service != "GloVe":
                # Initialize LanceDB on first call
                embeddings_db = EmbeddingsDB()
                embeddings_db.connect()
                if embeddings_db.db is None:
                    logger.error("Database object is not properly initialized")
                    return
                table_row_count = 0
                table_name = (
                    embedding_service.lower().strip().replace(" ", "_") + "_embeddings"
                )
                if table_name not in embeddings_db.db.table_names():
                    st.warning(
                        f"Beep Boop, table '{table_name}' does not exist. Please ingest data first."
                    )

                else:
                    table_row_count = embeddings_db.retrieve_table_row_count(table_name)
                    if table_row_count == 0:
                        st.warning(
                            "Beep Boop, Nothing found in the database. Please ingest data first."
                        )
                    else:
                        st.write(
                            f"Beep Boop, Found {table_row_count} words in the database."
                        )

                ## User can ingest more data into the database
                add_data_result = False
                with st.expander("Ingest more words into database"):
                    # Display a form, user can input number of words to embeddings into database.
                    with st.form("Ingest words into database:"):
                        num_words = st.number_input(
                            label="Number of words to ingest",
                            min_value=1,
                            max_value=1000,
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
                    st.rerun()

            st.subheader("Enter words with operations (- or +) between them")
            st.caption("Example: king - man + woman")
            st.caption("Press Enter when done.")

            with st.form("Input words and operations", enter_to_submit=True):
                analogy_input = st.text_input(
                    label="Input words and operations", value=""
                )
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
                        similar_words = get_similar_words_from_db(
                            embeddings_db,
                            embedding_service,
                            target_word_embedding,
                            words,
                        )
                        if similar_words:
                            display_similar_words(similar_words)
                        else:
                            st.warning("Beep Boop, SBERT can't find any similar words.")
                    else:  # Cohere, OpenAI, or Azure OpenAI
                        similar_words = get_similar_words_from_db(
                            embeddings_db,
                            embedding_service,
                            target_word_embedding,
                            words,
                        )
                        if similar_words:
                            display_similar_words(similar_words)
                        else:
                            st.warning(
                                "Could not find any suitable answers in the vocabulary."
                            )


if __name__ == "__main__":
    main()
