# ==========================================================
# 🧪 LOADING VERSION PRO — Tests métier clairs et pertinents
# ==========================================================
# These tests focus only on meaningful, production-ready behavior.
# They reflect how a real Data/AI team validates critical functions:
# readable, targeted, and centered on business logic.
# ==========================================================
# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts.cleaning import RENAME_MAPPING
from scripts.loading import READ_DEFAULT, loading_df

# ==========================================================
# 🔎 TEST — loading_df : valid file
# ==========================================================

def test_loading_df_single_file(mocker):

    mocker.patch("os.path.exists", return_value=True)


    mock_df = pd.DataFrame({
        "Animal": ["A"],
        "Country": ["X"],
        "Length_cm": [10],
        "Date": ["2024-01-01"],
    })

    mocker.patch("pandas.read_csv", return_value=mock_df)

    df = loading_df(["fake.csv"])

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 4)  # 1 ligne, 4 colonnes


# ==========================================================
# 🔎 TEST — loading_df : file not found
# ==========================================================

def test_loading_df_file_not_found(mocker):
    mocker.patch("os.path.exists", return_value=False)

    with pytest.raises(FileNotFoundError):
        loading_df(["missing.csv"])


# ==========================================================
# 🔎 TEST — loading_df : missing columns (warning)
# ==========================================================

def test_loading_df_missing_columns_warning(mocker):
    mocker.patch("os.path.exists", return_value=True)

    mock_df = pd.DataFrame({"A": [1]})

    read_csv = mocker.patch("pandas.read_csv", return_value=mock_df)
    logger = mocker.patch("scripts.loading.logger")

    df = loading_df(["fake.csv"])

    assert logger.warning.called


# ==========================================================
# 🔎 TEST — loading_df : concaténation multi-files
# ==========================================================

def test_loading_df_multiple_files_concat(mocker):
    mocker.patch("os.path.exists", return_value=True)

    df1 = pd.DataFrame({"Animal": ["A"], "Country": ["X"]})
    df2 = pd.DataFrame({"Animal": ["B"], "Country": ["Y"]})

    mocker.patch("pandas.read_csv", side_effect=[df1, df2])

    df = loading_df(["file1.csv", "file2.csv"])

    assert df.shape[0] == 2
    assert set(df["Animal"]) == {"A", "B"}
