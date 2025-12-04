# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from configuration.logger_config import get_logger
from utils.save_utils import save_model, save_stats

# ----------------------------------------------------------
# ⚙️ LOGGING
# ----------------------------------------------------------
logger = get_logger("gb_multioutput")
logger.info("🚀 Multi-output Gradient Boosting script initialized.")

# ----------------------------------------------------------
# 📥 LOAD DATA
# ----------------------------------------------------------
DATA_PATH = "data/cleaned/clean_animal_data.csv"

VALID_SPECIES = ["Red Squirrel", "Hedgehog", "European Bison", "Lynx"]

try:
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["Animal", "Country", "Length_cm", "Weight_kg"])
    df = df[df["Animal"].isin(VALID_SPECIES)]

    logger.info(f"✅ Dataset loaded — shape: {df.shape}")
except Exception as e:
    logger.error(f"❌ Error loading dataset: {e}")
    raise


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text

# ----------------------------------------------------------
# 🧩 FEATURES & TARGETS
# ----------------------------------------------------------
FEATURES = ["Animal", "Country"]
TARGETS = ["Length_cm", "Weight_kg"]

X_raw = df[FEATURES].copy()
Y = df[TARGETS].copy()

X = pd.get_dummies(X_raw, columns=["Animal", "Country"], drop_first=True)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# 🌲 MULTI-OUTPUT GRADIENT BOOSTING
# ----------------------------------------------------------
base_gb = GradientBoostingRegressor(random_state=42)
model = MultiOutputRegressor(base_gb)

model.fit(X_train, Y_train)

save_model(model, f"gradient_boosting/gradient_boosting_multioutput_animal_country_to_length_weight.pkl")

logger.info(f"💾 Multi-output model saved ")

model.feature_names_in_ = X.columns.to_numpy()

# ----------------------------------------------------------
# 📊 METRICS 
# ----------------------------------------------------------
Y_pred = model.predict(X_test)
Y_pred = np.asarray(Y_pred)   

# Safety: ensure dense Y
if hasattr(Y_test, "toarray"):
    Y_test = pd.DataFrame(Y_test.toarray(), columns=TARGETS)

metrics = []
for i, target_name in enumerate(TARGETS):
    y_true = Y_test.iloc[:, i]
    y_hat = Y_pred[:, i]

    mse = mean_squared_error(y_true, y_hat)
    mae = mean_absolute_error(y_true, y_hat)
    r2 = r2_score(y_true, y_hat)

    logger.info(
        f"[{target_name}] MSE={mse:.2f} | MAE={mae:.2f} | R²={r2:.2f}"
    )

    metrics.append(
        {
            "Target": target_name,
            "MSE": round(mse, 3),
            "MAE": round(mae, 3),
            "R²": round(r2, 3),
        }
    )

# ----------------------------------------------------------
# 💾 SAVE GLOBAL STATS
# ----------------------------------------------------------
df_stats = pd.DataFrame(metrics)
save_stats(df_stats, "ML/gradient_boosting/stats_gb_multioutput_length_weight.csv")

logger.info("📊 Multi-output stats saved successfully.")
