# ==========================================================
# 🧩 TEST DATA_UTILS FULL  — brille dans Codecov ✨
# ==========================================================
# This version covers all branches, errors, and edge cases
# to ensure complete code coverage. It is ideal for static
# analysis tools (Codecov) and overall project quality:
# exhaustive by design and intentionally detailed.
# ==========================================================
# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
import pytest

import utils.data_utils as du
from utils.data_utils import parse_date, safe_parse, stats

# ==========================================================
# 🔥 FULL COVERAGE 
# ==========================================================

@pytest.mark.parametrize(
    "inp, expected",
    [
        ("31/12/2025", "31/12/2025"),
        ("2021/04/04", "04/04/2021"),
        ("28-03-2000", "28/03/2000"),
        ("3 avril 2019", "03/04/2019"),
        ("février 2025", "01/02/2025"),
        ("01/01/22", "01/01/2022"),
        ("01/01/69", "01/01/1969"),
        ("31/12/99", "31/12/1999"),
        ("01/01/00", "01/01/2000"),
        ("2011", "01/01/2011"),
        ("23-12-25", "23/12/2025"),
        ("11 novembre 2021", "11/11/2021"),
        ("févr. 2024", "01/02/2024"),
        ("20 Mars 2006", "20/03/2006"),
        ("le 3 mai 2010", "03/05/2010"),
        ("mardi 7 janvier 1997", "07/01/1997"),
        ("03-18-2015", "18/03/2015"),
        ("2011/04/29", "29/04/2011"),
        ("03.13.2024", "13/03/2024"),
        ("1999", "01/01/1999"),
    ]
)
def test_parse_date_full_valid(inp, expected):
    assert parse_date(inp) == expected


@pytest.mark.parametrize(
    "inp",
    [
        "12/07",
        "2025/13/01",
        "32 janvier 2025",
        "10 décembre",
        "???",
        " ",
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
        None,
    ]
)
def test_parse_date_full_invalid(inp):
    assert parse_date(inp) is None


# ==========================================================
# 🔁 safe_parse 
# ==========================================================
def test_safe_parse_success():
    stats["success"] = 0
    stats["fail"] = 0

    def good(x):
        return x * 2
    assert safe_parse(good, 3) == 6
    assert stats["success"] == 1
    assert stats["fail"] == 0


def test_safe_parse_failure(caplog):
    stats["success"] = 0
    stats["fail"] = 0

    def bad(_):
        raise ValueError("boom")
    result = safe_parse(bad, 1, label="testfail")
    assert result is None
    assert stats["fail"] == 1
    assert "testfail failed" in caplog.text

def test_parse_date_invalid_month(monkeypatch):
    class FakeDate:
        year = 2020
        month = 13 
        day = 10
    monkeypatch.setattr(du, "parse", lambda *args, **kwargs: FakeDate)
    assert du.parse_date("2020-10-10") is None

def test_parse_date_invalid_day_combo(monkeypatch):
    class FakeDate:
        year = 2024
        month = 2
        day = 31  
    monkeypatch.setattr(du, "parse", lambda *args, **kwargs: FakeDate)
    assert du.parse_date("31 février 2024") is None

