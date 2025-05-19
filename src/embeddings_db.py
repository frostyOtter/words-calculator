import lancedb
import numpy as np
import os
from typing import List, Dict, Any
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

    def _ensure_db_directory(self) -> None:
        """Ensure the database directory exists."""
        if not os.path.exists(self.uri):
            os.makedirs(self.uri)

    def connect(self) -> None:
        """Connect to the LanceDB database."""
        self.db = lancedb.connect(self.uri)
        logger.info("Welcome to LanceDB!")

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
        results = (
            table.search(query_vector).distance_type("cosine").limit(limit).to_list()
        )
        return results
