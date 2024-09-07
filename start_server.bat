@echo off

if not exist venv (
    echo Virtual environment not found, creating one...
    python -m venv venv
)

echo Activating the virtual environment...
call venv\Scripts\activate

echo Installing required packages from requirements.txt...
pip install -r requirements.txt

echo Starting the FastAPI server...
python -m uvicorn webui.main:app --reload --host 0.0.0.0 --port 7777
