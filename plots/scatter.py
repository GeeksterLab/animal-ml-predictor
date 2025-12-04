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
# 🌐 Scatter + Mean Line — Weight by Gender by Animal
# ==========================================================   
def plot_scatter_line(df: pd.DataFrame):

    df = df.dropna(subset=["Gender", "Weight_kg", "Animal"])
    df = df[df["Gender"].isin(["Male", "Female"])]

    species_list = sorted(df["Animal"].unique())
    n = len(species_list)

    cols = 2
    rows = (n + 1) // cols

    plt.figure(figsize=(14, 4 * rows))

    for i, species in enumerate(species_list, 1):
        sub = df[df["Animal"] == species]

        mean_values = (
            sub.groupby("Gender", as_index=False)["Weight_kg"]
               .mean()
        )

        plt.subplot(rows, cols, i)

        sns.scatterplot(
            data=sub,
            x="Gender",
            y="Weight_kg",
            alpha=0.4,
            s=60,
            edgecolor="k",
            linewidth=0.3
        )

        sns.lineplot(
            data=mean_values,
            x="Gender",
            y="Weight_kg",
            marker="o",
            color="red",
            lw=2.5
        )

        plt.title(f"{species} — Weight by Gender", fontsize=12)
        plt.ylabel("Weight (kg)")
        plt.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("📊 Weight Distribution by Gender — per Species", fontsize=15, color="dimgray")
    plt.tight_layout()

    save_figure("visualization/scatter_weight_gender_by_species.png")


if __name__ == "__main__":
    logger.info("🚀 Starting species scatter generation...")
    df = pd.read_csv("data/cleaned/clean_animal_data.csv")
    plot_scatter_line(df)
    logger.info("✅ Done.")
