# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
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
# 📊 Barplot — Number of Animals per Country
# ==========================================================   

def plot_bar_count(df: pd.DataFrame):
    df = df.dropna(subset=["Country"])
    

    plt.figure(figsize=(12, 7), dpi=100)
    
    sns.countplot(
        data=df,
        x="Country",
        hue=None,    
        palette="Spectral" 
    )

    plt.xlabel("Country")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.suptitle("Where the Animals Are Located", fontsize=15, color="dimgray") 
    plt.tight_layout()

    save_figure("visualization/barplot_animals_by_country.png")

if __name__ == "__main__":
    logger.info("🚀 Starting species scatter generation...")
    df = pd.read_csv("data/cleaned/clean_animal_data.csv")
    plot_bar_count(df)
    logger.info("✅ Done.")
