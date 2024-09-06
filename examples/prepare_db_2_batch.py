from engine import dataset
from engine import engine
from engine import database
from engine.logging_config import log
from tqdm.rich import tqdm


batch_size = 64
# ImageLoader and DataLoader setup
il = dataset.ImageLoader("./dataset/flora_images", batch_size=batch_size)
dl = il.init_dataloader()

# EmbeddingGenerator and Database setup
eg = engine.EmbeddingGenerator(
    model_name="ViT-B-32", pretrained_model="laion2b_s34b_b79k"
)
db = database.Database("database/nature", "flora_20k_multi", 512)
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
