# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
import logging
import os
import re
import unicodedata

import pandas as pd

from configuration.logger_config import get_logger
from utils.data_utils import parse_date

# ==========================================================
# ⚙️ CONFIG LOGGING
# ==========================================================
logger = get_logger("cleaning")
logger.info("🚀 Cleaning script initialized successfully.")

# ==========================================================
# 🔧 UTILITAIRES GÉNÉRIQUES
# ==========================================================
def _normalize_string(val: str) -> str | None:
    """Normalise a string while preserving extended accented characters.

    Behaviour expectations (see tests):
    - Collapse multiple spaces → single space
    - Trim leading/trailing whitespace
    - Preserve unicode letters (including characters like "ĉ", "å", "é")
    - Preserve digits, spaces, dots and dashes
    - Drop other symbols
    - Return ``None`` for empty/whitespace-only input
    """
    if not isinstance(val, str):
        return None

    v = unicodedata.normalize("NFC", str(val)).replace("\u00A0", " ")
    v = v.strip().lower()

    v = re.sub(r"[^\w\s.-]", " ", v, flags=re.UNICODE)

    # Collapse multiple spaces created by replacements
    v = re.sub(r"\s+", " ", v).strip()

    return v or None

# ==========================================================
# 🧹 CLEANING PIPELINE FUNCTIONS
# ==========================================================

RENAME_MAPPING = {
    "Animal_type": "Animal",
    "Observation_date": "Date",
    "Body_Length_cm": "Length_cm",
}

def clean_dataset_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    columns_to_drop = [
        "Animal_code", "Animal_name", "Data_compiled_by"
    ]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors="ignore")

    if RENAME_MAPPING:
        df = df.rename(columns=RENAME_MAPPING, errors="ignore")

    return df


def cleaning_pipeline(Animal, Date, Country, Weight_kg, Length_cm, Gender, Latitude, Longitude) -> dict[str, str | float | int | None] | None:
    if pd.isna(Animal) and pd.isna(Date) and pd.isna(Country) and pd.isna(Weight_kg) and pd.isna(Length_cm) and pd.isna(Gender) and pd.isna(Latitude) and pd.isna(Longitude): return None

    cleaned : dict[str, str | float | int | None] = {}

    # ==========================================================
    #  COUNTRY
    # ==========================================================
    clean_country = _normalize_string(Country) if isinstance(Country, str) else None
    if clean_country:
        if clean_country in ["pl"]:
            clean_country = "poland"
        elif clean_country in ["hu", "hungry"]:
            clean_country = "hungary"
        elif clean_country in ["cc", "cz"]:
            clean_country = "czech republic"
        elif clean_country in ["de"]:
            clean_country = "germany"
    cleaned["Country"] = clean_country.strip().title() if clean_country else None

    # ==========================================================
    #  ANIMAL
    # ==========================================================
    clean_animal = _normalize_string(Animal) if isinstance(Animal, str) else None
    if clean_animal:
        for type in ["?", "™"]:
            clean_animal = clean_animal.replace(type, "")
        for name in ["busson", "bisson"]:
            clean_animal = clean_animal.replace(name, "bison")
        for hedge in ["ledghod", "wedgehod", "ledgheod", "ledgehod"]:
            clean_animal = clean_animal.replace(hedge, "hedgehog")
        for squir in ["squirrell", "squirlel", "squirel"]:
            clean_animal = clean_animal.replace(squir, "squirrel")

    cleaned["Animal"] = clean_animal.strip().title() if clean_animal else None

    cleaned["Date"] = parse_date(Date) 
    cleaned["Weight_kg"] = Weight_kg if isinstance(Weight_kg,(int,float)) else None
    cleaned["Length_cm"] = Length_cm if isinstance(Length_cm,(int,float)) else None
    cleaned["Gender"] = Gender.strip().title() if isinstance(Gender,str) else None
    cleaned["Latitude"] = Latitude if isinstance(Latitude,(int,float)) else None
    cleaned["Longitude"] = Longitude if isinstance(Longitude,(int,float)) else None

    return cleaned

# ==========================================================
# 📈 STATS
# ==========================================================
def afficher_statistiques(df: pd.DataFrame, logger=logging):
    """Display or log key dataframe statistics based on execution mode."""
    logger.info("========== 🧹 STATISTICS  ==========")
    logger.info(f"📏 Shape: {df.shape}")
    logger.info(f"🔑 Unique values (top10): {df.nunique().head(10).to_dict()}")
    logger.info(f"♻️ Duplicates: {int(df.duplicated().sum())}")
    logger.info(f"❓ Missing values (top10): {df.isna().sum().head(10).to_dict()}")

    logger.info(f"Bad/NA Animal: {df['Animal'].isna().sum() if 'Animal' in df.columns else '-'}")
    logger.info(f"Bad/NA Country: {df['Country'].isna().sum() if 'Country' in df.columns else '-'}")
    logger.info(f"Bad/NA Weight_kg: {df['Weight_kg'].isna().sum() if 'Weight_kg' in df.columns else '-'}")
    logger.info(f"Bad/NA Length: {df['Length'].isna().sum() if 'Length' in df.columns else '-'}")
    logger.info(f"Bad/NA Date: {df['Date'].isna().sum() if 'Date' in df.columns else '-'}")
    logger.info(f"Bad/NA Gender: {df['Gender'].isna().sum() if 'Gender' in df.columns else '-'}")
    logger.info(f"Bad/NA Longitude: {df['Longitude'].isna().sum() if 'Longitude' in df.columns else '-'}")
    logger.info(f"Bad/NA Latitude: {df['Latitude'].isna().sum() if 'Latitude' in df.columns else '-'}")
    logger.info("✅ Production inspection complete.")


