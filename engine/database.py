from typing import Any

import lancedb
import pyarrow as pa


class Database:
    """
    A class to manage connections and operations with LanceDB.

    Parameters
    ----------
    uri : str
        URI of the LanceDB instance.
    table_name : str
        Name of the table to create or connect to.
    embedding_size : int
        Size of the embedding vector for the schema.

    Methods
    -------
    get_table() -> Any
        Returns the connected or created table object.
    """

    def __init__(self, uri: str, table_name: str, embedding_size: int):
        """Initializes the Database connection, schema, and table."""
        self.db = lancedb.connect(uri)
        self.table_name = table_name
        self.schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), embedding_size)),
                pa.field("img_path", pa.string()),
            ]
        )
        self.tbl = self.db.create_table(
            self.table_name, schema=self.schema, exist_ok=True
        )

    def get_table(self) -> Any:
        """Returns the connected or created table object."""
        return self.tbl
