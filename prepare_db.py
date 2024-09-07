import yaml
from engine import dataset
from engine import engine
from engine import database
from engine.logging_config import log
from tqdm.rich import tqdm

# Load configuration from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Retrieve settings from config
batch_size = config["dataset"]["batch_size"]
image_folder = config["dataset"]["images_path"]
db_path = config["database"]["path"]
table_name = config["database"]["table_name"]
embedding_dim = config["database"]["embedding_dim"]
model_name = config["model"]["name"]
pretrained_model = config["model"]["pretrained_model"]

# ImageLoader and DataLoader setup
il = dataset.ImageLoader(image_folder, batch_size=batch_size)
dl = il.init_dataloader()

# EmbeddingGenerator and Database setup
eg = engine.EmbeddingGenerator(model_name=model_name, pretrained_model=pretrained_model)
db = database.Database(db_path, table_name, embedding_dim)
tbl = db.get_table()

for img_paths, images in tqdm(dl, desc="Processing Batches"):
    try:
        embeddings = eg.generate_image_embeddings(images, batch_size=batch_size)
    except Exception as e:
        log.error(f"Error generating embeddings for batch: {str(e)}")
        continue

    batch_data = [
        {"vector": embedding, "img_path": img_path}
        for embedding, img_path in zip(embeddings, img_paths)
    ]

    if batch_data:
        try:
            tbl.add(batch_data)
        except Exception as e:
            log.error(f"Error writing batch data to the database: {str(e)}")
