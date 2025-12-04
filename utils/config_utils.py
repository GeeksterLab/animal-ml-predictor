import os
import sys
from pathlib import Path

import pandas as pd

from configuration.logger_config import get_logger

# ==========================================================
# ⚙️ CONFIG LOGGING 
# ==========================================================
logger = get_logger("config_utils")

logger.propagate = True
logger.info("🚀 Configuration paths initialized.")

# ----------------------------------------------------------
# 🛤️ PATH
# ----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------
# 📦 DATA
# ----------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
DATA_CLEAN = DATA_DIR / "cleaned"
DATA_RAW = DATA_DIR / "raw"

AUTOFIX_INPUT = DATA_RAW / "animal_data_dirty.csv"
AUTOFIX_OUTPUT = DATA_RAW / "animal_data_dirty_reworked.csv"

DEFAULT_RAW_DATASET = DATA_RAW / "animal_data_dirty_reworked.csv"
DEFAULT_MAIN_DATASET = DATA_CLEAN / "clean_animal_data.csv"

# ----------------------------------------------------------
# 📊 EDA / PLOTS / STATS
# ----------------------------------------------------------
RESULTS_DIR = BASE_DIR / "results"

PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_EDA = PLOTS_DIR / "EDA"

STATS_DIR = RESULTS_DIR / "stats"
STATS_EDA = STATS_DIR / "EDA"

# ----------------------------------------------------------
# 🤖 MODELS
# ----------------------------------------------------------
MODEL_DIR = RESULTS_DIR / "model" / "ML"

GB_DIR = MODEL_DIR / "gradient_boosting"
KMEANS_DIR = MODEL_DIR / "kmeans"
FEATURE_DIR = RESULTS_DIR / "feature" / "gradient_boosting"

# ----------------------------------------------------------
# 📄 MODEL METADATA 
# ----------------------------------------------------------
METADATA_DIR = RESULTS_DIR / "metadata"
METADATA_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------
# 🔎 README / UTILS (if needed)
# ----------------------------------------------------------
README_PATH = BASE_DIR / "README.md"


# ==========================================================
# 🧠 UTILS
# ==========================================================
def csv_exists():
    return DEFAULT_MAIN_DATASET.exists()


def list_species_from_csv():
    # If the dataset does not exist, gracefully return an empty list
    if not DEFAULT_MAIN_DATASET.exists():
        return []

    df = pd.read_csv(DEFAULT_MAIN_DATASET)

    # If the expected column is not present, also return an empty list
    if "Animal" not in df.columns:
        return []

    species = df["Animal"].dropna().unique()

    # Ensure we always return a plain Python list, sorted alphabetically
    return sorted(species.tolist())

