import re
from datetime import datetime
from typing import Optional, Union

import pandas as pd
from dateparser import parse

from configuration.logger_config import get_logger

# ==========================================================
# ⚙️ CONFIG LOGGING
# ==========================================================
logger = get_logger("data_utils")

logger.propagate = True
logger.info("🚀 Data utils script initialized.")

# ==========================================================
# 🧩 WRAPPER — SAFE PARSE (Global Success/Fail Tracker)
# ==========================================================
# Track global parser stats
stats = {"success": 0, "fail": 0}

def safe_parse(parser, value, label=None):
    """
    Universal wrapper to safely execute any parser with centralized stats tracking.
    Access stats via:
        from utils.data_utils_template import stats
        print(stats)  # {'success': 950, 'fail': 50}
    """
    try:
        result = parser(value)
        stats["success"] += 1
        return result
    except Exception as e:
        stats["fail"] += 1
        logger.warning(f"⚠️ {label or parser.__name__} failed for value {value}: {e}")
        return None

# ==========================================================
# 📅 DATE FORMAT
# ==========================================================

def parse_date(last_date: Union[str, int, float, None]) -> Optional[str]:
    """Parse detailed 'last observation' date to DD/MM/YYYY format.
    """
    if last_date in ["NaN", "inconnu", "inconnue", "", "unknown", "n/a", "na"] or pd.isna(last_date):
        return None

    try:
        original = str(last_date).replace("\u00A0", " ").strip()
        observation = original.lower()

        FALLBACK_DATE_FORMATS = [
            # Y/M/D and D/M/Y with slashes or dashes
            "%y/%m/%d",
            "%Y/%m/%d",
            "%d/%m/%y",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%Y-%m-%d",
            "%y-%m-%d",
            # Support US-like MM-DD-YYYY and MM.DD.YYYY (but **not** MM/DD/YYYY)
            "%m-%d-%Y",
            "%m-%d-%y",
            "%d.%m.%Y",
            "%d.%m.%y",
            "%m.%d.%Y",
            "%m.%d.%y",
        ]

        date_obj = None
        if not observation:
            return None

        # Basic sanity check for explicit YYYY/MM/DD strings
        if re.match(r"^\d{4}/\d{1,2}/\d{1,2}$", observation):
            try:
                y, m, d = map(int, observation.split("/"))
                if not (1 <= m <= 12 and 1 <= d <= 31):
                    logger.warning(f"Invalid date values: {last_date}")
                    return None
            except ValueError: # pragma: no cover
                pass

        # Year only (e.g. "2011")
        if re.match(r"^\d{4}$", observation):
            date_obj = datetime.strptime(observation + "-01-01", "%Y-%m-%d")
        else:
            # First attempt: dateparser with French + English context
            date_obj = parse(
                observation,
                languages=["fr", "en"],
                settings={"DATE_ORDER": "DMY", "REQUIRE_PARTS": ["year"], "PREFER_DAY_OF_MONTH": "first"},
            )

        # Fallback: explicit strptime patterns
        for fmt in FALLBACK_DATE_FORMATS:
            if not date_obj:
                try:
                    date_obj = datetime.strptime(original, fmt)
                    break
                except Exception as e:
                    logger.debug(f"Failed with format {fmt}: {e}")
                    continue

        if date_obj:
            # Final consistency checks on month/day values
            if date_obj.month > 12 or date_obj.month < 1:
                logger.warning(f"Invalid date → incorrect month → {last_date}")
                return None
            try:
                datetime(date_obj.year, date_obj.month, date_obj.day)
            except ValueError:
                logger.warning(f"Invalid day/month combo: {last_date}")
                return None

        if not date_obj:
            logger.warning(f"Invalid date: {last_date}. Conversion failed.")
            return None

        return date_obj.strftime("%d/%m/%Y")

    except Exception as e: # pragma: no cover
        logger.warning(f"❌ Error converting date: {last_date} → {e}")
        return None
