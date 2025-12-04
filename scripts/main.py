# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
import argparse
import os
import time

import pandas as pd

from configuration.logger_config import get_logger
from scripts.cleaning import (
    afficher_statistiques,
    clean_dataset_base,
    cleaning_pipeline,
)
from scripts.loading import loading_df
from utils.save_utils import save_clean

# ==========================================================
# ⚙️ CONFIG LOGGING 
# ==========================================================
logger = get_logger("main")
logger.info("🚀 Main pipeline initialized.")

# ==========================================================
# LAMBDA PIPELINE
# ==========================================================
def main():
    """
    LAMBDA pipeline:
    Cleans dataset row by row (dynamic cleaning).
    Ideal for flexible or mixed-format datasets like AI content engagement.
    """

    start = time.time()
    logger.info("🤖 Starting LAMBDA pipeline...")

    try:
        df_raw = loading_df()
        logger.info(f"📥 {len(df_raw)} raw rows loaded.")

        df_structured = clean_dataset_base(df_raw)
        logger.info(f"📦 Base structure cleaned → {len(df_structured)} rows.")

        df_clean = df_structured.apply(lambda row: cleaning_pipeline(**row), axis=1)
        df_clean = pd.DataFrame(df_clean.tolist())        

        logger.info(f"✅ {df_clean.shape[0]} rows cleaned, {df_clean.shape[1]} validated columns.")

        save_clean(df_clean, "clean_animal_data.csv")
        afficher_statistiques(df_clean)

    except Exception as e:
        logger.exception(f"❌ Error in LAMBDA pipeline: {e}")

    finally:
        duration = time.time() - start
        logger.info(f"🏁 LAMBDA pipeline completed in {duration:.2f} seconds.")

if __name__ == "__main__":
    main()




