#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Virtual environment not found, creating one..."
    python3 -m venv venv
fi

echo "Activating the virtual environment..."
source venv/bin/activate

echo "Installing required packages from requirements.txt..."
pip install -r requirements.txt

echo "Starting the FastAPI server..."
python3 -m uvicorn webui.main:app --reload --host 0.0.0.0 --port 7777
