import logging
import sys
from pathlib import Path
from typing import Union, Optional

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"

def setup_logger(
    name: str, 
    log_file: Optional[Union[str, Path]] = None, 
    level: int = logging.INFO,
    stream: bool = True
) -> logging.Logger:
    """
    Sets up a logger with optional file and stream handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers if setup is called multiple times
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(LOG_FORMAT)

    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
