import lancedb
import matplotlib.pyplot as plt
from engine import engine
from PIL import Image

db = lancedb.connect("./database/nature/")
db.table_names()

tbl = db.open_table("flora_20k_multi")
tbl.count_rows()

df = tbl.to_pandas()

eg = engine.EmbeddingGenerator(
    model_name="ViT-B-32", pretrained_model="laion2b_s34b_b79k"
)

img_embed = eg.generate_image_embedding_from_path(img)

neigh = tbl.search(img_embed).limit(5).to_pandas()
