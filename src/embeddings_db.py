import lancedb
import numpy as np
import os
from typing import List, Dict, Any, Callable
from loguru import logger


class EmbeddingsDB:
    def __init__(self, uri: str = "data/lancedb"):
        """Initialize the EmbeddingsDB with a database URI.

        Args:
            uri (str): Path to the LanceDB database directory
        """
        self.uri = uri
        self.db = None
        self._ensure_db_directory()

    # Decorator to ensure the table name ends with '_embeddings'
    @staticmethod
    def ensure_table_name(func):
        """Decorator to ensure the first argument 'table_name' ends with '_embeddings'."""

        def wrapper(self, table_name, *args, **kwargs):
            if not table_name.endswith("_embeddings"):
                table_name = table_name + "_embeddings"
            return func(self, table_name, *args, **kwargs)

        return wrapper

    def _ensure_db_directory(self) -> None:
        """Ensure the database directory exists."""
        if not os.path.exists(self.uri):
            os.makedirs(self.uri)

    def connect(self) -> None:
        """Connect to the LanceDB database."""
        self.db = lancedb.connect(self.uri)
        logger.info("Beep Boop, Welcome to LanceDB!")

    @ensure_table_name
    def open_table(self, table_name: str) -> Any:
        """Open an existing table.

        Args:
            table_name (str): Name of the table to open

        Returns:
            The opened table object

        Raises:
            ValueError: If the table doesn't exist
        """
        if not self.db:
            self.connect()

        if table_name not in self.db.table_names():
            raise ValueError(f"Table '{table_name}' does not exist")

        return self.db.open_table(table_name)

    @ensure_table_name
    def create_table(self, table_name: str, data: List[Dict[str, Any]]) -> Any:
        """Create a new table with the given data.

        Args:
            table_name (str): Name of the table to create
            data (List[Dict[str, Any]]): List of dictionaries containing the data
                                        Each dict should have a 'vector' key with the embedding

        Returns:
            The created table object

        Raises:
            ValueError: If the table already exists
        """
        if not self.db:
            self.connect()

        if table_name in self.db.table_names():
            raise ValueError(f"Table '{table_name}' already exists")

        return self.db.create_table(table_name, data=data)

    @ensure_table_name
    def retrieve_table_row_count(self, table_name: str) -> int:
        """Retrieve the number of rows in the specified table.

        Args:
            table_name (str): Name of the table to retrieve information about

        """
        if not self.db:
            self.connect()

        if table_name not in self.db.table_names():
            raise ValueError(f"Table '{table_name}' does not exist")

        return len(self.db.open_table(table_name))

    @ensure_table_name
    def search(
        self, table_name: str, query_vector: np.ndarray, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in the specified table.

        Args:
            table_name (str): Name of the table to search in
            query_vector (np.ndarray): The query vector to search for
            limit (int): Maximum number of results to return

        Returns:
            List[Dict[str, Any]]: List of search results

        Raises:
            ValueError: If the table doesn't exist
        """
        if not self.db:
            self.connect()

        if table_name not in self.db.table_names():
            raise ValueError(f"Table '{table_name}' does not exist")

        table = self.open_table(table_name)
        logger.info("Beep Boop, Searching...")
        try:
            results = (
                table.search(query_vector)
                .distance_type("cosine")
                .limit(limit)
                .to_list()
            )
            return results
        except Exception as e:
            logger.error(f"Beep Boop, Error: {e}")
            return []

    @ensure_table_name
    def add_new_data_to_table(
        self, table_name: str, embeddings_model: Callable, data: List[str]
    ) -> bool:

        embeddings_data = embeddings_model.generate_embeddings(data)
        logger.debug(f"Beep Boop, Generated: {len(embeddings_data)} embeddings")
        data = [
            {"text": text, "vector": vector}
            for text, vector in zip(data, embeddings_data)
        ]

        logger.debug(f"Beep Boop, Adding data to table: {table_name}")
        # check if table exists
        try:
            if table_name not in self.db.table_names():
                logger.warning(
                    f"Table '{table_name}' does not exist. Creating table..."
                )
                self.create_table(table_name, data)
            else:
                table = self.open_table(table_name)
                table.add(data)
        except Exception as e:
            logger.error(f"Error adding data to table: {e}")
            return False
        return True
