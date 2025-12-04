# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from configuration.logger_config import get_logger
from utils.save_utils import save_stats

# =============================
# ⚙️ CONFIG LOGGING 
# =============================
logger = get_logger("plot")

logger.info("🚀 Plots script initialized.")

# ==============================
# 📊 Table - Descriptive Stats
# =============================

def generate_descriptive_stats(df: pd.DataFrame):

    # ==========================
    # 🧮 1. Numerical variables
    # ==========================
    num_df = df.select_dtypes(include=["int64", "float64"])

    def count_outliers(series: pd.Series) -> int:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return int(((series < lower) | (series > upper)).sum())

    numerical_stats = pd.DataFrame({
        "mean": num_df.mean(),
        "median": num_df.median(),
        "std": num_df.std(),
        "var": num_df.var(),
        "min": num_df.min(),
        "25%": num_df.quantile(0.25),
        "75%": num_df.quantile(0.75),
        "max": num_df.max(),
        "skew": num_df.skew(),
        "kurtosis": num_df.kurtosis(),
        "missing_%": num_df.isna().mean() * 100,
        "outliers": num_df.apply(count_outliers)
    })

    numerical_stats = numerical_stats.round(3)

    # ==========================
    # 🎭 2. Categorical variables
    # ==========================
    cat_df = df.select_dtypes(include=["object"])

    categorical_stats = pd.DataFrame({
        "unique": cat_df.nunique(),
        "mode": cat_df.mode().iloc[0],
        "mode_freq": cat_df.apply(lambda x: x.value_counts().max()),
        "missing_%": cat_df.isna().mean() * 100
    })

    categorical_stats = categorical_stats.sort_index()

    # ==========================
    # 📦 3. Retour formaté
    # ==========================
    return {
        "numerical_stats": numerical_stats,
        "categorical_stats": categorical_stats
    }


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned/clean_animal_data.csv")

    stats = generate_descriptive_stats(df)

    print("\n=== 📊 Numerical Stats ===")
    print(stats["numerical_stats"])

    print("\n=== 🎭 Categorical Stats ===")
    print(stats["categorical_stats"])
    
    with open("results/stats/EDA/eda_numerical_stats.md", "w") as f:
        f.write(stats["numerical_stats"].to_markdown())

    with open("results/stats/EDA/eda_categorical_stats.md", "w") as f:
        f.write(stats["categorical_stats"].to_markdown())

    # # Option CVS
    # stats["numerical_stats"].to_csv("results/stats/numerical_stats.csv")
    # stats["categorical_stats"].to_csv("results/stats/categorical_stats.csv")