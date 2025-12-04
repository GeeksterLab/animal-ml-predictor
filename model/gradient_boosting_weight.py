# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from configuration.logger_config import get_logger
from utils.save_utils import save_feature, save_model, save_stats, save_train

# ----------------------------------------------------------
# ⚙️ CONFIG LOGGING
# ----------------------------------------------------------
logger = get_logger("gradientboosting")
logger.info("🚀 Gradient Boosting modeling script initialized successfully.")

# ----------------------------------------------------------
# 📥 DATA LOADING
# ----------------------------------------------------------
DATA_PATH = "data/cleaned/clean_animal_data.csv"

VALID_SPECIES = ["Red Squirrel", "Hedgehog", "European Bison", "Lynx"]

try:
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["Weight_kg", "Length_cm", "Animal"])

    df = df[df["Animal"].isin(VALID_SPECIES)]

    logger.info(f"✅ Dataset loaded successfully — shape: {df.shape}")

except Exception as e:
    logger.error(f"❌ Error loading dataset: {e}")
    raise

# ----------------------------------------------------------
# 🧩 UTILS
# ----------------------------------------------------------
def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text

# ----------------------------------------------------------
# 🌲 MODELING PER SPECIES
# ----------------------------------------------------------
stats = []
feature_importances = []

FEATURES = ["Length_cm"]   

for species, df_subset in df.groupby("Animal"):

    if len(df_subset) < 10:
        logger.warning(f"⚠️ Not enough data for species '{species}' ({len(df_subset)} rows)")
        continue

    y = df_subset["Weight_kg"]
    X = df_subset[FEATURES]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    save_model(model, f"gradient_boosting/gradient_boosting_weight_{slugify(species)}.pkl")

    # Feature importances
    importances = model.feature_importances_
    current_features = X.columns.tolist()

    for feature_name, importance in zip(current_features, importances):
        feature_importances.append({
            "Species": species,
            "Feature": feature_name,
            "Importance": round(float(importance), 4)
        })

    # Performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=min(len(X), 5), scoring="r2")
    cv_mean = np.mean(cv_scores)

    logger.info(f"[{species}] MSE={mse:.2f} | MAE={mae:.2f} | R²={r2:.2f} | CV_R²_mean={cv_mean:.2f}")

    stats.append({
        "Animal": species,
        "n": len(df_subset),
        "MSE": round(mse, 2),
        "MAE": round(mae, 2),
        "R²": round(r2, 2),
        "CV_R²_mean": round(cv_mean, 2)
    })

    # Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, label="Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal fit")
    plt.xlabel("Weight — Actual (kg)")
    plt.ylabel("Weight — Predicted (kg)")
    plt.title(f"{species} (Gradient Boosting) → MSE: {mse:.2f} | R²: {r2:.2f}")
    plt.legend()
    plt.tight_layout()

    save_train(plt, f"gradient_boosting/gradientboosting_weight_{slugify(species)}.png")

# Save stats
df_stats = pd.DataFrame(stats)
save_stats(df_stats, "ML/gradient_boosting/stats_gradientboosting_weight_per_species.csv")
logger.info("📊 GradientBoosting per Animal — stats saved successfully.")

# Save feature importances
df_feature = pd.DataFrame(feature_importances)
save_feature(df_feature, "gradient_boosting/feature_importance_gradientboosting_weight_per_species.csv")

logger.info("📊 GradientBoosting feature importances saved successfully.")
