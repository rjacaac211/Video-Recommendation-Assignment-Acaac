import os
import json
import logging
from datetime import datetime

# Set up basic logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FILE = "../logs/application.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=LOG_FORMAT)

def log(message, level="info"):
    """
    Logs a message with the specified level.
    Args:
        message (str): Message to log.
        level (str): Log level ('info', 'warning', 'error').
    """
    log_levels = {
        "info": logging.info,
        "warning": logging.warning,
        "error": logging.error
    }
    log_function = log_levels.get(level, logging.info)
    log_function(message)

def load_config(file_path):
    """
    Loads configuration from a JSON file.
    Args:
        file_path (str): Path to the configuration JSON file.
    Returns:
        dict: Configuration data.
    """
    if not os.path.exists(file_path):
        log(f"Configuration file not found: {file_path}", level="error")
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, "r") as f:
        config = json.load(f)
    log(f"Configuration loaded from {file_path}")
    return config

def timestamp():
    """
    Get the current timestamp in ISO format.
    Returns:
        str: Current timestamp as a string.
    """
    return datetime.now().isoformat()

def save_json(file_path, data):
    """
    Saves data to a JSON file.
    Args:
        file_path (str): Path to the JSON file.
        data (dict): Data to save.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    log(f"Data saved to {file_path}")

def load_json(file_path):
    """
    Loads data from a JSON file.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        dict: Loaded data.
    """
    if not os.path.exists(file_path):
        log(f"File not found: {file_path}", level="error")
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)
    log(f"Data loaded from {file_path}")
    return data
