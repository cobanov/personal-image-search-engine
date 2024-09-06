> ## ⚠️ **Warning:** This project is under heavy development and may undergo significant changes.

# Personal Image Search Engine

## To-Do

- [ ] Find a good name for the project
- [x] Multiprocessing image reading
- [ ] Multigpu embedding calculation
- [x] Save embedddings embeddings folder, create if not exists
- [x] FastAPI interface
- [x] Web app
- [ ] dockerfile
- [ ] Face Engine
- [x] Migrate DB to LanceDB
- [ ] Load imagepaths from filelist.txt
- [ ] Logging should be seperate
- [ ] Lancdb on webapp
- [ ] I don't think batch processing works properly, use torch stack

## Benchmarks

- _20k images 25min on 3090ti 06.09.2024_

## Tasks

- [] Test save utils.save_image_paths_as_csv()

## Installation

```
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Docker

```bash
docker build -t personal-image-search .
docker run -p 7777:7777 personal-image-search
```

or

```bash
docker-compose up --build
```

## Usage

```bash
python -m uvicorn webui.main:app --reload --host 0.0.0.0 --port 7777
```

## Known Problems

### relative and absolute paths for image dataset and image paths

```python
app.mount(
    "/dataset",
    StaticFiles(
        directory="C:/Users/hope/Desktop/developer/personal-image-search-engine/dataset"
    ),
    name="dataset",
)

```

### embedding files are not in the correct place

```python
index_file_path = r"C:\Users\hope\Desktop\developer\personal-image-search-engine\embeddings\random_images_faiss.index"
file_paths_path = r"C:\Users\hope\Desktop\developer\personal-image-search-engine\embeddings\random_images_paths.csv"
```

### lib40ml problem

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

### relative imports are mess

```python
sys.path.append(os.path.abspath("../engine"))

from utils import read_image

from engine import EmbeddingGenerator, QueryEngine
```

## Creating Image Database

```python
import utils
from embedding_model import EmbeddingGenerator

# Initialize the model
eg = EmbeddingGenerator()

# Generate embeddings from folder
image_folder = "./dataset/random"
image_paths = utils.read_images_from_directory(image_folder)
imgs = utils.read_with_pil(image_paths)

# Generate embeddings from images
img_embeddings = eg.generate_image_embeddings(imgs)

# Save the embeddings, file paths and faiss index
eg.save_embeddings_as_npy(img_embeddings, "embeddings/random_images_embeddings.npy")
eg.save_file_paths(image_paths, "embeddings/random_images_paths.csv")
eg.save_faiss_index(img_embeddings, "embeddings/random_images_faiss.index")
```

## Query Image

```python
import os

import pandas as pd

import embedding_model
import utils
from plotting import plot_list_of_images

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Initialize the embedding model
query_engine = embedding_model.QueryEngine()

# Load the faiss index
query_engine.load_faiss_index("./embeddings/faiss_index.index")

# Query an image
source_image = r"dataset\random\2023-04-02_15-15-53_UTC.jpg"
file_paths = "embeddings/file_paths.csv"

# Load the file paths
file_paths = pd.read_csv(file_paths)["file_path"]
img = utils.read_image(source_image)

# Search for the nearest neighbors
distances, indices = query_engine.search_images(img, 4)
nearest_neighbors_paths = [file_paths[i] for i in indices[0]]

# Plot the source image and the nearest neighbors
plot_list_of_images(source_image, nearest_neighbors_paths)
```

### Query Text

```python
import os

import pandas as pd

import embedding_model
import utils
from plotting import plot_list_of_images

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Initialize the embedding model
query_engine = embedding_model.QueryEngine()

# Load the faiss index
query_engine.load_faiss_index("./embeddings/faiss_index.index")

# Query an image
text_query = r"a puppy dog"
file_paths = "embeddings/file_paths.csv"

# Load the file paths
file_paths = pd.read_csv(file_paths)["file_path"]

# Search for the nearest neighbors
distances, indices = query_engine.search_text(text_query, 4)
nearest_neighbors_paths = [file_paths[i] for i in indices[0]]

# Plot the source image and the nearest neighbors
plot_list_of_images(None, nearest_neighbors_paths)
```
