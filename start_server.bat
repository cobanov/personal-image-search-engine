@echo off

:: Check if the virtual environment directory exists
if not exist venv (
    echo Virtual environment not found, creating one...
    python -m venv venv
)

:: Activate the virtual environment
echo Activating the virtual environment...
call venv\Scripts\activate

:: Install required packages if necessary
echo Installing required packages from requirements.txt...
pip install -r requirements.txt

:: Start the FastAPI app using uvicorn
echo Starting the FastAPI server...
python -m uvicorn webui.main:app --reload --host 0.0.0.0 --port 7777
