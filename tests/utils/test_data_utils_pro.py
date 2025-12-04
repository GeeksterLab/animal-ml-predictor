# ==========================================================
# 🧩 🧪 DATA_UTILS VERSION PRO — Tests métier clairs et pertinents
# ==========================================================
# These tests focus only on meaningful, production-ready behavior.
# They reflect how a real Data/AI team validates critical functions:
# readable, targeted, and centered on business logic.
# ==========================================================
# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
import pytest

from utils.data_utils import parse_date


# ==========================================================
# 🔹 TESTS PRO — Most representative cases
# ==========================================================
@pytest.mark.parametrize(
    "inp, expected",
    [
        ("31/12/2025", "31/12/2025"),
        ("2021/04/04", "04/04/2021"),
        ("28-03-2000", "28/03/2000"),
        ("3 avril 2019", "03/04/2019"),
        ("février 2025", "01/02/2025"),
        ("23-12-25", "23/12/2025"),
        ("11 novembre 2021", "11/11/2021"),
        ("févr. 2024", "01/02/2024"),
        ("20 Mars 2006", "20/03/2006"),
        ("le 3 mai 2010", "03/05/2010"),
        ("mardi 7 janvier 1997", "07/01/1997"),
        ("2011", "01/01/2011"),
    ]
)
def test_parse_date_valid_pro(inp, expected):
    """Tests métier représentatifs — version PRO."""
    assert parse_date(inp) == expected


# ==========================================================
# 🔹 TESTS PRO — Critical invalid cases
# ==========================================================
@pytest.mark.parametrize(
    "inp",
    [
        "12/07",
        "2025/13/01",
        "32 janvier 2025",
        "10 décembre",
        "???",
        "2025/aa/10",
        "2025/00/10",
        "2025/02/30",
        "99/99/9999",
        "32/06/2015",
        "09/13/2015",  
        "2025/00/aa",
        "aa/00/01",
        "--",
        "..",
        "###",
        "abc123",
        "",
        " ",
        None,
    ]
)
def test_parse_date_invalid_pro(inp):
    assert parse_date(inp) is None
