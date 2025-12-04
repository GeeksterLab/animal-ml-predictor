# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
import logging
import os
from datetime import datetime

from colorama import Fore, Style
from colorama import init as colorama_init

colorama_init(autoreset=True)

def get_logger(name: str = "default") -> logging.Logger:


    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{name}_logs_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  

    logger.setLevel(logging.INFO)
    logger.propagate = False  

    # ==========================================================
    # 1️⃣ Handler FICHIER
    # ==========================================================
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "⌚️ %(asctime)s - 📌 %(levelname)s - 💬 %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # ==========================================================
    # 2️⃣ Handler CONSOLE 
    # ==========================================================
    class ColorFormatter(logging.Formatter):
        """Custom console formatter with color-coded log levels."""

        COLORS = {
            "DEBUG": Fore.BLUE,
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.MAGENTA + Style.BRIGHT,
        }

        def format(self, record):
            color = self.COLORS.get(record.levelname, "")
            message = super().format(record)
            return f"{color}{message}{Style.RESET_ALL}"

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColorFormatter("📢 %(levelname)s — %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"🚀 Logger '{name}' initialized (logs → {log_path})")

    return logger

