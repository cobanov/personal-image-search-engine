## ⚠️ **Warning:** This project is under active development and may undergo significant changes.

# Personal Image Search Engine

## To-Do

- [ ] Find a suitable name for the project.
- [x] Enable multiprocessing for image reading.
- [ ] Implement multi-GPU support for embedding calculations.
- [x] Save embeddings in a designated folder (create if not exists).
- [x] Implement a FastAPI-based interface.
- [x] Build a web app.
- [x] Create a Dockerfile for easy deployment.
- [ ] Integrate Face Detection Engine.
- [x] Migrate the database to LanceDB.
- [ ] Load image paths from `filelist.txt`.
- [x] Separate logging functionality for better modularity.
- [x] Integrate LanceDB with the web app.
- [x] Fix batch processing in embedding calculations using `torch.stack()`.

## Benchmarks

- **Performance**: _20k images processed in 13 minutes on an NVIDIA 3090 Ti (as of 06.09.2024)._

## Installation

To get started, install the required packages:

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Docker

You can build and run the project using Docker:

```bash
docker build -t personal-image-search .
docker run -p 7777:7777 personal-image-search
```

Alternatively, using Docker Compose:

```bash
docker-compose up --build
```

## Usage

### Running the FastAPI app using scripts

#### Windows

Run the following script:

```bash
start_server.bat
```

#### Linux/macOS

First, ensure the script has executable permissions:

```bash
chmod +x start_server.sh
```

Then run it:

```bash
./start_server.sh
```

## Known Issues

### lib40ml Problem

In case you encounter issues related to `lib40ml`, you can resolve them by setting the following environment variable in your code:

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
