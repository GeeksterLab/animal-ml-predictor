# ==========================================================
# 🧪 MAIN VERSION PRO — Tests métier clairs et pertinents
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

from scripts.main import main

# ==========================================================
# 🔎 TEST — main() 
# ==========================================================

def test_main_runs_nominal(mocker):

    mocker.patch("scripts.main.loading_df",
                 return_value=pd.DataFrame({"Animal": ["A"], "Country": ["X"]}))


    mocker.patch("scripts.main.clean_dataset_base",
                 return_value=pd.DataFrame({"Animal": ["A"], "Country": ["X"]}))


    mocker.patch("scripts.main.cleaning_pipeline",
                 return_value={"Animal": "A", "Country": "X"})

    # Mock save_clean 
    mock_save = mocker.patch("scripts.main.save_clean")

    # Mock afficher_statistiques
    mock_stats = mocker.patch("scripts.main.afficher_statistiques")

    result = main()

    assert result is None   
    assert mock_save.called
    assert mock_stats.called


# ==========================================================
# 🔎 TEST — main() gère les exceptions
# ==========================================================

def test_main_exception_handling(mocker):

    mocker.patch("scripts.main.loading_df",
                 side_effect=RuntimeError("Boom"))

    mock_log = mocker.patch("scripts.main.logger")

    result = main()

    assert mock_log.exception.called
    assert result is None
