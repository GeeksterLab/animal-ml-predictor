# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from configuration.logger_config import get_logger
from utils.save_utils import save_feature, save_model, save_stats, save_train

# ----------------------------------------------------------
# ⚙️ CONFIG LOGGING
# ----------------------------------------------------------
logger = get_logger("linearregression")
logger.info("🚀 Linear Regression modeling script initialized successfully.")

# ----------------------------------------------------------
# 📥 DATA LOADING
# ----------------------------------------------------------
DATA_PATH = "data/cleaned/clean_animal_data.csv"

VALID_SPECIES = ["Red Squirrel", "Hedgehog", "European Bison", "Lynx"]

try:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Length_cm", "Weight_kg", "Latitude", "Longitude", "Animal", "Country"])
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
# 📊 LINEAR REGRESSION PER SPECIES
# ----------------------------------------------------------
stats = []  # store per-Animal results
feature_importances = [] # store feature importances used

FEATURES = ["Weight_kg", "Country"]

for species, df_subset in df.groupby("Animal"):
    if len(df_subset) < 10:
        logger.warning(f"⚠️ Not enough data for Animal '{species}' ({len(df_subset)} rows)")
        continue

    y = df_subset[["Length_cm"]]
    X = df_subset[FEATURES]
    X = pd.get_dummies(X, columns=['Country'], drop_first=True)

    # --- Standardisation ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === Split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # === Model ===
    model = LinearRegression()
    model.fit(X_train, y_train)

    # === Save trained model ===
    save_model(model, f"linear_regression/linearregression_{slugify(species)}.pkl")

    # === FEATURE IMPORTANCE (COEFFICIENT) ===
    current_features = X.columns.tolist()
    for feature_name, coef in zip(current_features, model.coef_[0]):
        feature_importances.append({
            "Species": species,
            "Feature": feature_name,
            "Importance": round(float(coef), 4)
        })

    # === Predictions ===
    y_pred = model.predict(X_test)

    # === Metrics ===
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=min(len(X), 10), scoring="r2")
    cv_mean = np.mean(cv_scores)

    logger.info(f"[{species}] MSE={mse:.2f} | R²={r2:.2f} | CV_R²_mean={cv_mean:.2f}")

    # === Save stats row ===
    stats.append({
        "Animal": species,
        "n": len(df_subset),
        "MSE": round(mse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 2),
        "CV_R²_mean": round(cv_mean, 2)
    })

    # === Plot: Actual vs Predicted ===
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, label="Samples")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--", label="Ideal fit")
    plt.xlabel("Length — Actual (cm)")
    plt.ylabel("Length — Predicted (cm)")
    plt.title(f"{species} (LinearRegression) → MSE: {mse:.2f} | R²: {r2:.2f}",fontsize= 15, color="dimgray")
    plt.legend()
    plt.tight_layout()

    save_train(plt,
        f"linear_regression/linearregression_{slugify(species)}.png")

# ----------------------------------------------------------
# 💾 SAVE GLOBAL STATS
# ----------------------------------------------------------
df_stats = pd.DataFrame(stats)

save_stats(df_stats, "ML/linear_regression/stats_linearregression_per_species.csv")

logger.info("📊 LinearRegression per Animal — stats saved successfully.")

# ----------------------------------------------------------
# 💾 SAVE FEATURE IMPORTANCE FOR STREAMLIT
# ----------------------------------------------------------
df_feature = pd.DataFrame(feature_importances)

save_feature(df_feature, "linear_regression/feature_importance_linearregression_per_species.csv")

logger.info("📊 LinearRegression feature importances saved successfully.")