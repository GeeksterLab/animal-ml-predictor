# ==========================================================
# 🧩 TEST CLEANING FULL — brille dans Codecov ✨
# ==========================================================
# This version covers all branches, errors, and edge cases
# to ensure complete code coverage. It is ideal for static
# analysis tools (Codecov) and overall project quality:
# exhaustive by design and intentionally detailed.
# ==========================================================
# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
from unittest.mock import MagicMock

import pandas as pd
import pytest

from scripts.cleaning import (
    RENAME_MAPPING,
    _normalize_string,
    afficher_statistiques,
    clean_dataset_base,
    cleaning_pipeline,
)
from utils.data_utils import parse_date


def test_normalize_string_edge_cases():
    assert _normalize_string("   ") is None or _normalize_string("   ") == ""
    assert _normalize_string("a   b") == "a b"
    assert _normalize_string("ĉåfé") == "ĉåfé"
    assert _normalize_string("hello--world") == "hello--world"


def test_clean_dataset_base_empty_mapping():
    df = pd.DataFrame({"A": [1]})
    cleaned = clean_dataset_base(df)
    assert isinstance(cleaned, pd.DataFrame)


def test_cleaning_pipeline_country_variants():
    tests = {
        "pl": "Poland",
        "hu": "Hungary",
        "hungry": "Hungary",
        "cc": "Czech Republic",
        "cz": "Czech Republic",
        "de": "Germany",
    }

    for raw, expected in tests.items():
        result = cleaning_pipeline(
            Animal="A",
            Date="2024-01-01",
            Country=raw,
            Weight_kg=1,
            Length_cm=10,
            Gender="F",
            Latitude=1,
            Longitude=1,
        )
        assert result is not None 
        assert result["Country"] == expected


def test_cleaning_pipeline_invalid_types():
    result = cleaning_pipeline(
        Animal=123,
        Date=987,
        Country=True,
        Weight_kg="INVALID",
        Length_cm="BAD",
        Gender=999,
        Latitude="nope",
        Longitude=[],
    )

    assert result is not None 
    assert result["Animal"] is None
    assert result["Country"] is None
    assert result["Weight_kg"] is None
    assert result["Length_cm"] is None
    assert result["Gender"] is None
    assert result["Latitude"] is None
    assert result["Longitude"] is None


def test_cleaning_pipeline_partial_values():
    result = cleaning_pipeline(
        Animal="Bisson",   # test du remplacement bisson → bison
        Date="2024/01/02",
        Country=None,
        Weight_kg=5,
        Length_cm=None,
        Gender="M",
        Latitude=50.2,
        Longitude=None,
    )
    assert result is not None 
    assert result["Animal"] == "Bison"
    assert result["Date"] == parse_date("2024/01/02")
    assert result["Latitude"] == 50.2


def test_afficher_statistiques_full_branch(mocker):
    logger = mocker.MagicMock()

    df = pd.DataFrame({
        "Animal": [None],
        "Country": [None],
        "Weight_kg": [None],
        "Length": [None],
        "Date": [None],
        "Gender": [None],
        "Longitude": [None],
        "Latitude": [None],
    })

    afficher_statistiques(df, logger=logger)
    assert logger.info.call_count >= 8
