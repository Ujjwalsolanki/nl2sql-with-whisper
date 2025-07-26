# logger_config.py

import logging
import os
from datetime import datetime

def setup_logging(
    log_file_prefix: str = "nl2sql_app", # Changed to prefix
    log_level: str = "INFO",
    console_output: bool = True
) -> logging.Logger:
    """
    Configures and returns a logger for the application.
    Generates a new log file with a timestamp each time it's run.

    Args:
        log_file_prefix (str): The prefix for the log file name. A timestamp will be appended.
                               e.g., "nl2sql_app" will result in "nl2sql_app_YYYYMMDD_HHMMSS.log".
        log_level (str): The minimum logging level to capture (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        console_output (bool): Whether to also output logs to the console.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger("nl2sql_app")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Prevent duplicate handlers if called multiple times (important for Streamlit reruns)
    # Clear existing handlers to ensure fresh setup on each run/rerun
    if logger.handlers:
        for handler in logger.handlers[:]: # Iterate over a copy
            logger.removeHandler(handler)
        logger.handlers = [] # Clear the list after removal

    # Define a common formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log_file_prefix is provided
    if log_file_prefix:
        # Ensure the logs directory exists
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        # Generate a timestamp for the log file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"{log_file_prefix}_{timestamp}.log"
        file_path = os.path.join(log_dir, log_file_name)

        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        print(f"Logging configured. Level: {log_level.upper()}. Logs will be saved to '{file_path}' and/or console.")
    else:
        print(f"Logging configured. Level: {log_level.upper()}. Logs will be output to console only.")

    return logger