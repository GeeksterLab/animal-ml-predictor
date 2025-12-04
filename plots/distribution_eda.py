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
# 📊 Scatterplot - Weight vs Lengh
# ==========================================================

def plot_scatter(df: pd.DataFrame):
    df = df.dropna(subset=["Weight_kg", "Length_cm", "Gender"])

    plt.figure(figsize=(12, 8), dpi=100)

    sns.scatterplot(
        data=df,
        x="Weight_kg",
        y="Length_cm",
        hue="Gender",
        alpha=0.5,
        s=45,
    )

    plt.title("📈 EDA Length vs Weight — grouped by Gender", fontsize=14, pad=15)
    plt.xlabel("Weight (kg)")
    plt.ylabel("Length (cm)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Gender")
    plt.tight_layout()

    save_figure("EDA/distribution_weight_vs_lengh_by_gender.png")

if __name__ == "__main__":
    logger.info("🚀 Starting plot generation.")
    df = pd.read_csv("data/cleaned/clean_animal_data.csv")
    plot_scatter(df)
    logger.info("✅ End of plot run.")