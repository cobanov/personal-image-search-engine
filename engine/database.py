import lancedb
import pyarrow as pa


class Database:
    def __init__(self, uri, table_name, embedding_size):
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

    def get_table(self):
        return self.tbl
