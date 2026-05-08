# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
import logging
import os
import re
import unicodedata

import pandas as pd

from configuration.logger_config import get_logger

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

    v = unicodedata.normalize("NFC", str(val)).replace("\u00a0", " ")
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
    "Body_Length_cm": "Length_cm",
}


def clean_dataset_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    columns_to_drop = [
        "Animal_code",
        "Animal_name",
        "Data_compiled_by",
        "Observation_date",
    ]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors="ignore")

    if RENAME_MAPPING:
        df = df.rename(columns=RENAME_MAPPING, errors="ignore")

    return df


def cleaning_pipeline(
    Animal=None,
    Country=None,
    Weight_kg=None,
    Length_cm=None,
    Gender=None,
    Latitude=None,
    Longitude=None,
) -> dict[str, str | float | int | None] | None:
    if (
        pd.isna(Animal)
        and pd.isna(Country)
        and pd.isna(Weight_kg)
        and pd.isna(Length_cm)
        and pd.isna(Gender)
        and pd.isna(Latitude)
        and pd.isna(Longitude)
    ):
        return None

    cleaned: dict[str, str | float | int | None] = {}

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

    cleaned["Weight_kg"] = Weight_kg if isinstance(Weight_kg, (int, float)) else None
    cleaned["Length_cm"] = Length_cm if isinstance(Length_cm, (int, float)) else None
    cleaned["Gender"] = Gender.strip().title() if isinstance(Gender, str) else None
    cleaned["Latitude"] = Latitude if isinstance(Latitude, (int, float)) else None
    cleaned["Longitude"] = Longitude if isinstance(Longitude, (int, float)) else None

    return cleaned


# ==========================================================
# 📈 STATS
# ==========================================================
def afficher_statistiques(df: pd.DataFrame, logger=logging):
    """Display or log key dataframe statistics based on execution mode."""

    def _to_hashable(value):
        if isinstance(value, dict):
            return tuple(sorted((k, _to_hashable(v)) for k, v in value.items()))
        if isinstance(value, list):
            return tuple(_to_hashable(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(_to_hashable(v) for v in value))
        return value

    unique_values = (
        df.apply(lambda col: col.map(_to_hashable).nunique()).head(10).to_dict()
    )
    logger.info("========== 🧹 STATISTICS  ==========")
    logger.info(f"📏 Shape: {df.shape}")
    logger.info(f"🔑 Unique values (top10): {unique_values}")
    logger.info(f"♻️ Duplicates: {int(df.duplicated().sum())}")
    logger.info(f"❓ Missing values (top10): {df.isna().sum().head(10).to_dict()}")

    logger.info(
        f"Bad/NA Animal: {df['Animal'].isna().sum() if 'Animal' in df.columns else '-'}"
    )
    logger.info(
        f"Bad/NA Country: {df['Country'].isna().sum() if 'Country' in df.columns else '-'}"
    )
    logger.info(
        f"Bad/NA Weight_kg: {df['Weight_kg'].isna().sum() if 'Weight_kg' in df.columns else '-'}"
    )
    logger.info(
        f"Bad/NA Length: {df['Length'].isna().sum() if 'Length' in df.columns else '-'}"
    )
    logger.info(
        f"Bad/NA Gender: {df['Gender'].isna().sum() if 'Gender' in df.columns else '-'}"
    )
    logger.info(
        f"Bad/NA Longitude: {df['Longitude'].isna().sum() if 'Longitude' in df.columns else '-'}"
    )
    logger.info(
        f"Bad/NA Latitude: {df['Latitude'].isna().sum() if 'Latitude' in df.columns else '-'}"
    )
    logger.info("✅ Production inspection complete.")
