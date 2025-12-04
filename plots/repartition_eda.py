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
# 📊 Violonplot - Country vs Weight 
# ==========================================================
def plot_violinplot(df: pd.DataFrame):

    df = df.dropna(subset=["Country", "Weight_kg"])  

    plt.figure(figsize=(10, 8))
    sns.violinplot(
        data=df,
        x="Country",
        y="Weight_kg",
        hue=None,        
        palette="pastel",
        inner="quartile", 
        cut=0,
        scale="width"   
    )

    plt.title("🎻 Violinplot — Weight by Country", fontsize=14)
    plt.xlabel("Country")
    plt.ylabel("Weight (kg)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_figure("EDA/repartion_weight_by_country.png")

if __name__ == "__main__":
    logger.info("🚀 Starting plot generation.")
    df = pd.read_csv("data/cleaned/clean_animal_data.csv")
    plot_violinplot(df)
    logger.info("✅ End of plot run.")