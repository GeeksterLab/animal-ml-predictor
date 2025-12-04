from pathlib import Path

from configuration.logger_config import get_logger

# ==========================================================
# ⚙️ CONFIG LOGGING 
# ==========================================================
logger = get_logger("paths_utils")

logger.propagate = True
logger.info("🚀 Test save utils template initialized.")


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]
