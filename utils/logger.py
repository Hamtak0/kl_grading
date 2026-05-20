import logging
import sys
from pathlib import Path

def setup_logger(name: str ="KneeProject", log_file: str ="app.log") -> logging.Logger:
    """
    Sets up a logger that outputs to both the console and a file.
    """
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    log_file_path = logs_dir / log_file

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        return logger

    # Handlers
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter (Added %(name)s to see which script generated the log)
    formatter = logging.Formatter(
        '%(asctime)s - [%(name)s] - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger