import logging
import os
from datetime import datetime

from rich.logging import RichHandler

# Create logs folder if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a unique log file name based on the current timestamp
log_filename = os.path.join(
    log_dir, f"app_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",  # Format for file and console logs
    datefmt="%Y-%m-%d %H:%M:%S",  # Custom date format
    handlers=[
        RichHandler(
            show_time=False, show_level=False, show_path=False
        ),  # Console output via RichHandler
        logging.FileHandler(log_filename, mode="w"),  # Create a new file for every run
    ],
)

# Create a logger instance
log = logging.getLogger("rich")
