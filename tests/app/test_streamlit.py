import os
import pytest
import pandas as pd
import joblib

from pathlib import Path
from streamlit.testing.v1 import AppTest
from utils.config_utils import DEFAULT_MAIN_DATASET
from utils.config_utils import GB_DIR

APP_PATH = Path(__file__).resolve().parents[2] / "app" / "app.py"


def test_page_project_loads():
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=20)

    assert not app.exception
    assert any(
        "Animal Morphology Lab" in element.value
        for element in [*app.header, *app.markdown]
    )


def test_page_data_exploration_loads():
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=30)
    app.radio[0].set_value(":material/search: Data Exploration")
    app.run(timeout=30)

    assert not app.exception
    assert any("Data Exploration" in h.value for h in app.header)
    assert any("Raw Dataset Preview" in s.value for s in app.subheader)
    assert any("Clean Dataset Preview" in s.value for s in app.subheader)


def test_page_data_analysis_loads():
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=30)
    app.radio[0].set_value(":material/scatter_plot: Data Analysis")
    app.run(timeout=30)

    assert not app.exception
    assert any("Data Visualization" in h.value for h in app.header)


def test_page_modeling_prediction_loads():
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=30)
    app.radio[0].set_value(":material/psychology: Modeling & Prediction")
    app.run(timeout=40)

    assert not app.exception
    assert any("Modeling & Prediction" in h.value for h in app.header)
    assert any("Interactive Prediction" in h.value for h in app.header)


def test_prediction_button_does_not_crash():
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=30)
    app.radio[0].set_value(":material/psychology: Modeling & Prediction")
    app.run(timeout=40)

    assert not app.exception

    predict_buttons = [btn for btn in app.button if "Predict" in btn.label]

    assert predict_buttons, "Predict button not found."

    predict_buttons[0].click()
    app.run(timeout=40)

    assert not app.exception


def test_clean_dataset_structure():
    df = pd.read_csv(DEFAULT_MAIN_DATASET)

    expected_cols = [
        "Country",
        "Animal",
        "Weight_kg",
        "Length_cm",
        "Gender",
        "Latitude",
        "Longitude",
    ]

    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    assert df.isna().sum().sum() == 0, "Clean dataset still contains NaN"


def test_models_load():

    model_files = list(GB_DIR.glob("*.pkl"))

    assert model_files, "No model files found"

    for path in model_files:
        model = joblib.load(path)
        assert model is not None


def test_prediction_output():
    model = joblib.load(GB_DIR / "gradient_boosting_weight_european_bison.pkl")

    X = pd.DataFrame([{"Length_cm": 300}])

    pred = model.predict(X)

    assert pred is not None
    assert pred[0] > 0


def test_species_list_not_empty():
    df = pd.read_csv(DEFAULT_MAIN_DATASET)

    species_list = df["Animal"].dropna().unique()

    assert len(species_list) > 0


def test_expected_gradient_boosting_models_exist():
    from utils.config_utils import GB_DIR

    expected_models = [
        "gradient_boosting_length_european_bison.pkl",
        "gradient_boosting_weight_european_bison.pkl",
        "gradient_boosting_length_hedgehog.pkl",
        "gradient_boosting_weight_hedgehog.pkl",
        "gradient_boosting_length_lynx.pkl",
        "gradient_boosting_weight_lynx.pkl",
        "gradient_boosting_length_red_squirrel.pkl",
        "gradient_boosting_weight_red_squirrel.pkl",
    ]

    missing = [name for name in expected_models if not (GB_DIR / name).exists()]

    assert not missing, f"Missing model files: {missing}"


def test_expected_gradient_boosting_models_load():

    expected_models = [
        "gradient_boosting_length_european_bison.pkl",
        "gradient_boosting_weight_european_bison.pkl",
        "gradient_boosting_length_hedgehog.pkl",
        "gradient_boosting_weight_hedgehog.pkl",
        "gradient_boosting_length_lynx.pkl",
        "gradient_boosting_weight_lynx.pkl",
        "gradient_boosting_length_red_squirrel.pkl",
        "gradient_boosting_weight_red_squirrel.pkl",
    ]

    for name in expected_models:
        path = GB_DIR / name
        model = joblib.load(path)
        assert model is not None
        assert hasattr(model, "predict")
