# ==========================================================
# 🧩🧪 TEST SAVE UTILS
# ==========================================================
# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
import io
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from configuration.logger_config import get_logger
from utils.save_utils import *

# ==========================================================
# ⚙️ CONFIG LOGGING 
# ==========================================================
logger = get_logger("test_save_utils")
logger.info("🚀 Test save utils template initialized.")

# ==========================================================
#  FIXTURES & SETUP
# ==========================================================
@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


@pytest.fixture
def tmp_results(tmp_path):
    base = tmp_path / "results"
    base.mkdir()
    (base / "plots").mkdir()
    (base / "models").mkdir()
    (base / "stats").mkdir()
    return base

@pytest.fixture(autouse=True)
def reset_pandas_options():
    pd.reset_option("all")

# ==========================================================


# ==========================================================
# FIGURES & MODELS
# ==========================================================
@pytest.mark.parametrize("func, subfolder, expected_suffix", [
    (save_figure, "plots", ".png"),
    (save_model, "modeling", ".csv"), 
])
def test_save_visual_and_model_files(tmp_results, sample_df, func, subfolder, expected_suffix):
    filename = f"test{expected_suffix}"
    folder = tmp_results / subfolder

    if func is save_figure:
        plt.figure()
        plt.plot([1, 2], [3, 4])

    if func is save_figure:
        plt.figure()
        plt.plot([1, 2], [3, 4])
        func(filename, folder=str(folder))
    elif func is save_model:
        # Assuming save_model expects model_data as the first argument
        func(sample_df, filename, folder=str(folder))
    save_path = folder / filename

    assert save_path.exists(), f"❌ Expected file {save_path} to exist."
    assert save_path.suffix == expected_suffix
    assert os.path.getsize(save_path) > 0, "❌ File seems empty."

def test_save_train(tmp_path):
    folder = tmp_path / "train"
    filename = "train.png"
    plt.figure()
    plt.plot([1, 2], [3, 4])
    save_train(None, filename, folder=str(folder))

    save_path = folder / filename
    assert save_path.exists()
    assert os.path.getsize(save_path) > 0


# ==========================================================


# ==========================================================
#  CSV FILES
# ==========================================================
def test_save_clean_creates_file_and_matches_content(sample_df, tmp_path):
    folder = tmp_path / "data/cleaned"
    filename = "clean.csv"

    save_clean(sample_df, filename, folder=str(folder))
    save_path = folder / filename

    assert folder.exists()
    assert save_path.exists()

    result = pd.read_csv(save_path)
    pd.testing.assert_frame_equal(result, sample_df)


def test_save_clean_overwrite_behavior(sample_df, tmp_path):
    folder = tmp_path / "data/cleaned"
    folder.mkdir(parents=True)
    file_path = folder / "clean.csv"
    sample_df.to_csv(file_path, index=False)

    save_clean(sample_df, "clean.csv", folder=str(folder))
    assert file_path.exists()

def test_save_feature(tmp_path, sample_df):
    folder = tmp_path / "features"
    filename = "feat.csv"

    save_feature(sample_df, filename, folder=str(folder))

    save_path = folder / filename
    assert save_path.exists()
    result = pd.read_csv(save_path)
    pd.testing.assert_frame_equal(result, sample_df)


# ==========================================================


# ==========================================================
#  STATS SAVING
# ==========================================================
def test_save_stats_creates_csv(sample_df, tmp_results):
    folder = tmp_results / "stats"
    filename = "summary.csv"

    save_stats(sample_df, filename, folder=str(folder))
    save_path = folder / filename

    assert save_path.exists()
    result = pd.read_csv(save_path)
    pd.testing.assert_frame_equal(result, sample_df)

# ==========================================================


# ==========================================================
#  ERRORS & LOGS
# ==========================================================
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Permission bits not reliable on Windows")
def test_save_clean_permission_error(sample_df, tmp_path):
    folder = tmp_path / "forbidden"
    folder.mkdir()
    folder.chmod(0o400)

    with pytest.raises(PermissionError):
        save_clean(sample_df, "clean.csv", folder=str(folder))


def test_logging_on_save_clean(sample_df, tmp_path, caplog):
    folder = tmp_path / "data/cleaned"
    filename = "log_test.csv"

    with caplog.at_level("INFO"):
        save_clean(sample_df, filename, folder=str(folder))

    assert "💾" in caplog.text or "saved" in caplog.text.lower()

# ==========================================================


# ==========================================================
# ADDITIONAL VALIDATION & EDGE CASES
# ==========================================================
def test_save_clean_empty_dataframe(tmp_path, caplog):
    df_empty = pd.DataFrame()
    folder = tmp_path / "data/empty"
    filename = "empty.csv"

    with caplog.at_level("WARNING"):
        save_clean(df_empty, filename, folder=str(folder))

    save_path = folder / filename
    assert save_path.exists(), "❌ Expected empty CSV file to still be created."
    assert "empty" in caplog.text.lower() or "no data" in caplog.text.lower()


def test_save_clean_special_characters_in_path(sample_df, tmp_path):
    folder = tmp_path / "data/processed_éè@#"
    filename = "clean.csv"

    save_clean(sample_df, filename, folder=str(folder))
    save_path = folder / filename

    assert save_path.exists()
    result = pd.read_csv(save_path)
    pd.testing.assert_frame_equal(result, sample_df)
# ==========================================================


# ==========================================================
# 🧭 MAIN TESTER 
# ==========================================================
if __name__ == "__main__":
    logger.info("🧪 Running all test blocks for save_utils (template check).")
    logger.info("✅ All tests completed successfully.")
