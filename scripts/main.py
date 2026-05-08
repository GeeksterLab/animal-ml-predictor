# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
import argparse
import os
import time
from typing import Any
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

    start = time.time()
    logger.info("🤖 Starting LAMBDA pipeline...")

    try:
        df_raw = loading_df()
        logger.info(f"📥 {len(df_raw)} raw rows loaded.")

        df_structured = clean_dataset_base(df_raw)
        logger.info(f"📦 Base structure cleaned → {len(df_structured)} rows.")

        records = df_structured.to_dict(orient="records")

        cleaned_rows = []

        for record in records:
            cleaned = cleaning_pipeline(**{str(k): v for k, v in record.items()})

            if cleaned is not None:
                cleaned_rows.append(cleaned)

        df_clean = pd.DataFrame(cleaned_rows)

        REQUIRED_CLEAN_COLUMNS = [
            "Country",
            "Animal",
            "Weight_kg",
            "Length_cm",
            "Gender",
            "Latitude",
            "Longitude",
        ]

        present_required_columns = [
            col for col in REQUIRED_CLEAN_COLUMNS if col in df_clean.columns
        ]
        if present_required_columns:
            df_clean = df_clean.dropna(subset=present_required_columns)
        df_clean = df_clean.reset_index(drop=True)

        df_clean = pd.DataFrame(df_clean)
        df_clean = df_clean.dropna(how="all")
        df_clean = df_clean.drop_duplicates()

        logger.info(
            f"✅ {df_clean.shape[0]} rows cleaned, {df_clean.shape[1]} validated columns."
        )

        save_clean(df_clean, "clean_animal_data.csv")
        afficher_statistiques(df_clean)

    except Exception as e:
        logger.exception(f"❌ Error in LAMBDA pipeline: {e}")

    finally:
        duration = time.time() - start
        logger.info(f"🏁 LAMBDA pipeline completed in {duration:.2f} seconds.")


if __name__ == "__main__":  # pragma: no cover
    main()
