# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the FastAPI app runs on
EXPOSE 7777

# Define environment variable for FastAPI to run in reload mode and bind to all IP addresses
ENV HOST=0.0.0.0
ENV PORT=7777

# Command to run the FastAPI server
CMD ["uvicorn", "webui.main:app", "--reload", "--host", "0.0.0.0", "--port", "7777"]
