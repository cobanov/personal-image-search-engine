import os
from uuid import uuid4

import lancedb
from engine import EmbeddingGenerator
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Initialize router
router = APIRouter()

# Connect to LanceDB
db = lancedb.connect("./database/nature/")
tbl = db.open_table("flora_20k_multi")

# Initialize Embedding Generator
eg = EmbeddingGenerator(model_name="ViT-B-32", pretrained_model="laion2b_s34b_b79k")

templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=templates_dir)


@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        unique_filename = f"{uuid4()}_{file.filename}"
        file_location = f"uploaded_images/{unique_filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        image_embedding = eg.generate_image_embedding_from_path(file_location)
        neighbors = tbl.search(image_embedding).limit(24).to_pandas()
        neighbors_paths = neighbors["img_path"].to_list()

        return {"image_paths": neighbors_paths}
    except Exception as e:
        return {"error": f"Failed to upload or process image: {str(e)}"}


@router.post("/search/")
async def search(query: str = Form(...)):
    try:
        text_embedding = eg.generate_text_embedding(query)
        neighbors = tbl.search(text_embedding).limit(24).to_pandas()
        neighbors_paths = neighbors["img_path"].to_list()

        return {"image_paths": neighbors_paths}
    except Exception as e:
        return {"error": f"Failed to process search: {str(e)}"}
