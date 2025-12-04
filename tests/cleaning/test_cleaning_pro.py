# ==========================================================
# 🧪 CLEANING VERSION PRO — Tests métier clairs et pertinents
# ==========================================================
# These tests focus only on meaningful, production-ready behavior.
# They reflect how a real Data/AI team validates critical functions:
# readable, targeted, and centered on business logic.
# ==========================================================
# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------

from unittest.mock import MagicMock

import pandas as pd
import pytest

from scripts.cleaning import (
    _normalize_string,
    afficher_statistiques,
    clean_dataset_base,
    cleaning_pipeline,
)
from utils.data_utils import parse_date

# ==========================================================
# 🔎 TEST _normalize_string
# ==========================================================

def test_normalize_string_basic():
    assert _normalize_string("  HéLLô  ") == "héllô"


def test_normalize_string_removes_symbols():
    assert _normalize_string("Hello™@") == "hello"


def test_normalize_string_non_str_return_none():
    assert _normalize_string(123) is None
    assert _normalize_string(None) is None


# ==========================================================
# 🔎 TEST clean_dataset_base
# ==========================================================

def test_clean_dataset_base_drops_and_renames():
    df = pd.DataFrame({
        "Animal_type": ["A"],
        "Animal_code": [1],
        "Body_Length_cm": [100],
        "Animal_name": ["Zelda"],
        "Data_compiled_by": ["Me"]
    })

    cleaned = clean_dataset_base(df)

    assert "Animal_code" not in cleaned.columns
    assert "Animal_name" not in cleaned.columns
    assert "Data_compiled_by" not in cleaned.columns

    assert "Animal" in cleaned.columns
    assert "Length_cm" in cleaned.columns


# ==========================================================
# 🔎 TEST cleaning_pipeline
# ==========================================================

def test_cleaning_pipeline_valid_row():
    row = dict(
        Animal="squirrell™",
        Date="2024-01-02",
        Country="pl",
        Weight_kg=2.5,
        Length_cm=30,
        Gender="female",
        Latitude=10.0,
        Longitude=20.0,
    )

    result = cleaning_pipeline(**row)
    assert result is not None 
    assert result["Animal"] == "Squirrel"
    assert result["Country"] == "Poland"
    assert result["Weight_kg"] == 2.5
    assert result["Length_cm"] == 30
    assert result["Gender"] == "Female"
    assert result["Latitude"] == 10.0
    assert result["Longitude"] == 20.0
    assert result["Date"] == parse_date("2024-01-02")


def test_cleaning_pipeline_all_nan_returns_none():
    result = cleaning_pipeline(
        Animal=None, Date=None, Country=None,
        Weight_kg=None, Length_cm=None, Gender=None,
        Latitude=None, Longitude=None
    )
    assert result is None


def test_cleaning_pipeline_fix_common_typos():
    row = dict(
        Animal="ledgheod",  
        Date="2023-02-10",
        Country="hu",
        Weight_kg=1.2,
        Length_cm=15,
        Gender="male",
        Latitude=None,
        Longitude=None,
    )

    result = cleaning_pipeline(**row)
    assert result is not None 
    assert result["Animal"] == "Hedgehog"
    assert result["Country"] == "Hungary"


# ==========================================================
# 🔎 TEST afficher_statistiques (mock logging)
# ==========================================================

def test_afficher_statistiques_logs(mocker):
    logger = mocker.MagicMock()

    df = pd.DataFrame({
        "Animal": ["A", None],
        "Country": ["X", None],
        "Weight_kg": [1.0, None],
        "Length": [10, None],
        "Date": ["2024-01-01", None],
        "Gender": ["F", None],
        "Longitude": [1.0, None],
        "Latitude": [2.0, None],
    })

    afficher_statistiques(df, logger=logger)

    assert logger.info.call_count >= 5
