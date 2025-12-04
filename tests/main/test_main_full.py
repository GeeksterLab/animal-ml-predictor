# ==========================================================
# 🧩 TEST MAIN FULL  — brille dans Codecov ✨
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

from scripts.main import main


def test_main_full_success_flow(mocker):
    df_raw = pd.DataFrame({"Animal": ["A"], "Country": ["FR"]})
    df_struct = pd.DataFrame({"Animal": ["A"], "Country": ["FR"]})

    mocker.patch("scripts.main.loading_df", return_value=df_raw)
    mocker.patch("scripts.main.clean_dataset_base", return_value=df_struct)

    mocker.patch("scripts.main.cleaning_pipeline",
                 return_value={"Animal": "A", "Country": "France"})

    mock_save = mocker.patch("scripts.main.save_clean")
    mock_stats = mocker.patch("scripts.main.afficher_statistiques")
    mock_log = mocker.patch("scripts.main.logger")

    main()

    assert mock_log.info.call_count >= 3
    assert mock_save.called
    assert mock_stats.called


def test_main_full_empty_dataframe(mocker):
    df_empty = pd.DataFrame(columns=["Animal", "Country"])
    mocker.patch("scripts.main.loading_df", return_value=df_empty)

    mocker.patch("scripts.main.clean_dataset_base", return_value=df_empty)
    mocker.patch("scripts.main.cleaning_pipeline", return_value={})

    mock_save = mocker.patch("scripts.main.save_clean")
    mock_stats = mocker.patch("scripts.main.afficher_statistiques")

    main()

    assert mock_save.called
    assert mock_stats.called


def test_main_full_error_branch(mocker):
    mocker.patch("scripts.main.loading_df",
                 side_effect=ValueError("Error loading"))

    mock_log = mocker.patch("scripts.main.logger")
    main()

    assert mock_log.exception.called


def test_main_full_finally_branch(mocker):
    mocker.patch("scripts.main.loading_df",
                 side_effect=RuntimeError("Test error"))
    mock_log = mocker.patch("scripts.main.logger")

    main()

    assert any(
        "completed" in str(call.args[0]).lower()
        for call in mock_log.info.call_args_list
    )
