# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from configuration.logger_config import get_logger
from utils.save_utils import save_figure

# ==========================================================
# ⚙️ CONFIG LOGGING 
# ==========================================================
logger = get_logger("plot")

logger.info("🚀 Plots script initialized.")

# ==========================================================
# 📊 Heatmap - Latitude, Longitude, Weight and Length
# ==========================================================    
    
def plot_heatmap(df: pd.DataFrame):
    corr_matrix = df[["Latitude","Longitude","Weight_kg","Length_cm"]].dropna().corr(numeric_only=True)

    plt.figure(figsize=(12, 8), dpi=100)
    sns.heatmap(
        corr_matrix,
        cmap="YlGnBu", 
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=.5,
        linecolor="grey"
    )

    plt.title("🌡️ Global Correlation Map between Latitude, Longitude, Weight and Length", fontsize=14, pad=20)

    plt.suptitle(
        "Is there a Geographical Correlation between Latitude, Longitude, Weight and Length ?",
        fontsize=11,
        color="dimgray"
    )

    plt.tight_layout()
    save_figure("visualization/heatmap_lat_lon_weight_length.png")

if __name__ == "__main__":
    logger.info("🚀 Starting plot generation.")
    df = pd.read_csv("data/cleaned/clean_animal_data.csv")
    plot_heatmap(df)
    logger.info("✅ End of plot run.")