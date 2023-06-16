"""Logger utilities"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from tensorflow import get_logger


def init_logger(export_path: Path) -> None:
    """Initialize logger

    Parameters
    ----------
    export_path: Path
        Where save logs
    """
    logger = logging.getLogger("raido")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    logger_file = export_path / Path("logger.log")
    logger_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(logger_file, maxBytes=50 * 1024 * 1024, backupCount=10)
    # 50M file size with 10 backups = 0.5G logs :)
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    tf_logger = get_logger()
    ft_logger_file = export_path / Path("tensorflow.log")
    ft_fh = RotatingFileHandler(
        ft_logger_file, maxBytes=50 * 1024 * 1024, backupCount=10
    )
    ft_fh.setFormatter(formatter)
    # 50M file size with 10 backups = 0.5G logs :)
    tf_logger.setLevel(logging.INFO)
    tf_logger.addHandler(ft_fh)
