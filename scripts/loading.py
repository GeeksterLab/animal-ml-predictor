# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
import glob
import os
from typing import Iterable, List

import pandas as pd

from configuration.logger_config import get_logger
from scripts.cleaning import RENAME_MAPPING
from utils.config_utils import DEFAULT_RAW_DATASET

# ==========================================================
# ⚙️ CONFIG LOGGING 
# ==========================================================
logger = get_logger("loading")
logger.info("🚀 Loading script initialized.")


# ==========================================================
#  PARAMETERS & DEFAULT CONFIG
# ==========================================================
READ_DEFAULT = {
    "encoding": "utf-8-sig",
    "low_memory": False,
    "sep": ";",
}

# DEFAULT_FILE_PATH: List[str] = glob.glob(
#     "/Users/geeksterlab/Documents/PROJETS/animal_morphology_weight_predictor/data/raw/animal_data_dirty_reworked.csv"
# )

DEFAULT_FILE_PATH: List[str] = [str(DEFAULT_RAW_DATASET)]

# ==========================================================
# LOADER FUNCTION
# ==========================================================
def loading_df(file_path: Iterable[str] = DEFAULT_FILE_PATH, **read_csv_kwargs) -> pd.DataFrame:
    """
    Generic data loader for CSV-based datasets.
    Can handle multiple paths and automatically concatenate them.
    """
    dfs = []
    cfg = {**READ_DEFAULT, **read_csv_kwargs}

    for path in file_path:
        if os.path.exists(path):
            logger.info(f"📄 File found → {path}")
            try:
                df_temp = pd.read_csv(path, **cfg)
                df_temp.columns = df_temp.columns.str.strip()
                # --- Auto-detect expected columns from cleaning pipeline arguments ---
                expected_cols = set(RENAME_MAPPING.values())
                missing = expected_cols - set(df_temp.columns)
                if missing:
                    logger.warning(f"⚠️ Missing expected columns: {missing}")
                else:
                    logger.info("🟢 All required columns found.")
                dfs.append(df_temp)
                logger.info(f"✅ Successfully loaded: {os.path.basename(path)} ({df_temp.shape[0]} rows)")
            except Exception as e: # pragma: no cover
                logger.warning(f"⚠️ Reading error on {path}: {e}")
        else:
            logger.error(f"❌ File not found → {path}")

    if not dfs:
        raise FileNotFoundError(f"🔴 No valid file(s) found in paths: {file_path}")

    df_concat = pd.concat(dfs, ignore_index=True, copy=False)
    logger.info(f"🧩 {len(dfs)} file(s) loaded and concatenated → shape: {df_concat.shape}")
    return df_concat
