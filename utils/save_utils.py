import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from configuration.logger_config import get_logger

# ==========================================================
# ⚙️ CONFIG LOGGING 
# ==========================================================
logger = get_logger("save_utils")

logger.propagate = True
logger.info("🚀 Test save utils template initialized.")

# ----------------------------------------------------------
# 📦 BASE
# ----------------------------------------------------------
def ensure_save_path(folder: str, filename: str) -> str:
    save_path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return save_path
    
# ----------------------------------------------------------
# 🧹 PRE-PROCESSING
# ----------------------------------------------------------
def save_clean(df_clean: pd.DataFrame, filename: str, folder: str = "data/cleaned") -> None:
    save_path = ensure_save_path(folder, filename)
    df_clean.to_csv(save_path, index=False)

    # Log a more informative message depending on whether the DataFrame is empty
    if df_clean.empty:
        logger.warning(
            f"⚠️ Empty DataFrame — no data to save, created empty file at: {save_path}"
        )
    else:
        logger.info(f"💾 Cleaned DataFrame saved to : {save_path}")

# ----------------------------------------------------------
# 📊 EDA / PLOTS / STATS
# ----------------------------------------------------------
def save_figure(filename: str, folder: str = "results/plots") -> None:
    save_path = ensure_save_path(folder, filename)
    plt.savefig(save_path)
    logger.info(f"💾 Figure saved to {save_path}")
    plt.close()

def save_stats(df_stats, filename: str, folder: str = "results/stats") -> None:
    save_path = ensure_save_path(folder, filename)
    df_stats.to_csv(save_path, index=False)
    logger.info(f"💾 Stats file saved to : {save_path}")

# ----------------------------------------------------------
# 🤖 ML 
# ----------------------------------------------------------
def save_model(model, filename: str, folder: str = "results/model/ML") -> None:
    save_path = ensure_save_path(folder, filename)
    with open(save_path, 'wb') as file:
        pickle.dump(model, file)
    logger.info(f"💾 Model saved to : {save_path}")

def save_feature(df_feature: pd.DataFrame, filename: str, folder: str = "results/feature") -> None:
    save_path = ensure_save_path(folder, filename)
    df_feature.to_csv(save_path, index=False)
    logger.info(f"💾 Feature DataFrame saved to : {save_path}")

def save_train(df: pd.DataFrame, filename: str, folder: str = "results/modeling/ML") -> None:
    save_path = ensure_save_path(folder, filename)
    plt.savefig(save_path)
    logger.info(f"💾 Training saved to {save_path}")
    plt.close()