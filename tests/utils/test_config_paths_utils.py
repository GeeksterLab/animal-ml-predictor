# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
from pathlib import Path
from unittest.mock import patch

import pytest

import utils.config_utils as config_utils
from utils.paths_utils import get_project_root


# ----------------------------------------------------------
# 1. Tests for get_project_root()  (paths_utils.py)
# ----------------------------------------------------------
def test_get_project_root_return_type():
    root = get_project_root()
    assert isinstance(root, Path)


def test_get_project_root_is_parent_directory():
    root = get_project_root()
    utils_dir = Path(__file__).resolve().parents[1]
    assert root == utils_dir.parent.parent

# ----------------------------------------------------------
# 2. Tests for config_utils paths
# ----------------------------------------------------------
@pytest.mark.parametrize(
    "path_attr",
    [
        "BASE_DIR",
        "DATA_DIR",
        "DATA_CLEAN",
        "DATA_RAW",
        "DEFAULT_MAIN_DATASET",
        "RESULTS_DIR",
        "PLOTS_DIR",
        "PLOTS_EDA",
        "STATS_DIR",
        "STATS_EDA",
        "MODEL_DIR",
        "GB_DIR",
        "KMEANS_DIR",
        "FEATURE_DIR",
        "METADATA_DIR",
        "README_PATH",
    ]
)
def test_config_utils_paths_are_pathlib(path_attr):
    """All paths in config_utils must be Path objects."""
    assert isinstance(getattr(config_utils, path_attr), Path)


def test_metadata_dir_created():
    assert config_utils.METADATA_DIR.exists()
    assert config_utils.METADATA_DIR.is_dir()


# ----------------------------------------------------------
# 3. Tests for csv_exists()
# ----------------------------------------------------------
def test_csv_exists_true(tmp_path, monkeypatch):
    fake_csv = tmp_path / "clean_animal_data.csv"
    fake_csv.write_text("Animal,Weight_kg\nCat,4.0")

    monkeypatch.setattr(config_utils, "DEFAULT_MAIN_DATASET", fake_csv)

    assert config_utils.csv_exists() is True


def test_csv_exists_false(monkeypatch):
    fake_csv = Path("non_existing_file.csv")
    monkeypatch.setattr(config_utils, "DEFAULT_MAIN_DATASET", fake_csv)

    assert config_utils.csv_exists() is False


# ----------------------------------------------------------
# 4. Tests for list_species_from_csv()
# ----------------------------------------------------------
def test_list_species_from_csv_file_missing(monkeypatch):
    """If the CSV does not exist → return empty list."""
    monkeypatch.setattr(config_utils, "DEFAULT_MAIN_DATASET", Path("missing.csv"))
    assert config_utils.list_species_from_csv() == []


def test_list_species_from_csv_no_animal_column(tmp_path, monkeypatch):
    """If column Animal is missing → return empty list."""
    fake_csv = tmp_path / "dataset.csv"
    fake_csv.write_text("X,Y\n1,2")

    monkeypatch.setattr(config_utils, "DEFAULT_MAIN_DATASET", fake_csv)
    assert config_utils.list_species_from_csv() == []


def test_list_species_from_csv_ok(tmp_path, monkeypatch):
    fake_csv = tmp_path / "dataset.csv"
    fake_csv.write_text(
        "Animal,Weight_kg\n"
        "Lion,190\n"
        "Tiger,220\n"
        "Lion,200\n"
        "Zebra,300\n"
    )

    monkeypatch.setattr(config_utils, "DEFAULT_MAIN_DATASET", fake_csv)

    species = config_utils.list_species_from_csv()
    assert species == ["Lion", "Tiger", "Zebra"]  
