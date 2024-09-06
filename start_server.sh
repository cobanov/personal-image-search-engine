#!/bin/bash

# Check if the virtual environment directory exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found, creating one..."
    python3 -m venv venv
fi

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Install required packages if necessary
echo "Installing required packages from requirements.txt..."
pip install -r requirements.txt

# Start the FastAPI app using uvicorn
echo "Starting the FastAPI server..."
python3 -m uvicorn webui.main:app --reload --host 0.0.0.0 --port 7777
