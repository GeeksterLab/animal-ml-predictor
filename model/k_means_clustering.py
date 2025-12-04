# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from configuration.logger_config import get_logger
from utils.save_utils import save_clean, save_model, save_stats, save_train

# ----------------------------------------------------------
# ⚙️ CONFIG LOGGING
# ----------------------------------------------------------
logger = get_logger("kmeans")
logger.info("🚀 KMeans clustering script initialized.")

# ----------------------------------------------------------
# 📥 LOAD DATA
# ----------------------------------------------------------
DATA_PATH = "data/cleaned/clean_animal_data.csv"

VALID_SPECIES = ["Red Squirrel", "Hedgehog", "European Bison", "Lynx"]

try:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Weight_kg", "Length_cm"])
    df = df[df["Animal"].isin(VALID_SPECIES)]
    logger.info(f"📦 Dataset loaded — shape: {df.shape}")
except Exception as e:
    logger.error(f"❌ Error loading dataset: {e}")
    raise

# ----------------------------------------------------------
# 🧩 FEATURES
# ----------------------------------------------------------
FEATURES = ["Weight_kg", "Length_cm"]
X = df[FEATURES].copy()

# Standardisation (ESSENTIEL for KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------
# 🔍 AUTO-K : Elbow + Silhouette
# ----------------------------------------------------------
inertias = []
silhouettes = []
K_range = range(2, 10)

logger.info("🔎 Searching for best K using Elbow + Silhouette...")

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)

    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

best_k = K_range[np.argmax(silhouettes)]
logger.info(f"🌟 Best K selected by silhouette: K={best_k}")

# ----------------------------------------------------------
# 🔁 FINAL MODEL with best K
# ----------------------------------------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ----------------------------------------------------------
# 💾 SAVE MODEL
# ----------------------------------------------------------
save_model(kmeans, f"kmeans/kmeans_morpho_geo_k{best_k}.pkl")
save_model(scaler, "kmeans/kmeans_scaler.pkl")

# ----------------------------------------------------------
# 📦 SAVE CLUSTERED DATA
# ----------------------------------------------------------
save_clean(df, "ML/kmeans/data_with_clusters_kmeans.csv")

logger.info("💾 Clustered data saved with cluster assignments.")

# ----------------------------------------------------------
# 📊 SAVE STATS
# ----------------------------------------------------------
cluster_stats = df.groupby("Cluster")[FEATURES].mean().copy()
cluster_stats["count"] = df["Cluster"].value_counts().sort_index()

save_stats(cluster_stats, "ML/kmeans/kmeans_cluster_stats.csv")

logger.info("📊 Cluster stats saved.")

# ----------------------------------------------------------
# 🌍 PCA 2D — Visualisation des clusters dans l'espace réduit
# ----------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
components = pca.fit_transform(X_scaled)

df["PCA1"] = components[:, 0]
df["PCA2"] = components[:, 1]

# ----------------------------------------------------------
# 📈 PLOTS : Elbow + Silhouette
# ----------------------------------------------------------
# Elbow
plt.figure(figsize=(7, 5))
plt.plot(list(K_range), inertias, marker="o")
plt.title("Elbow Method — Inertia")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.suptitle("How Inertia Changes as We Increase K", y=0.95, fontsize=10, color ="gray")
plt.grid(True)

save_train(plt, "kmeans/elbow_kmeans.png")

# Silhouette
plt.figure(figsize=(7, 5))
plt.plot(list(K_range), silhouettes, marker="o", color="orange")
plt.title("Silhouette Score by K")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.suptitle("Which K Creates the Most Separated and Meaningful Clusters ?", y=0.95, fontsize=10, color ="gray")
plt.grid(True)

save_train(plt, "kmeans/silhouette_kmeans.png")

# Scatter final clusters (Weight vs Length)
plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=df,
    x="Weight_kg",
    y="Length_cm",
    hue="Cluster",
    palette="Spectral",
    alpha=0.8,
    s=70
)
plt.title("Final KMeans Clusters — Morphology (Weight vs Length)")
plt.suptitle("How Morphology Separates Animales Into Distinct Groups", y=0.98, fontsize=10, color ="gray")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

save_train(plt, "kmeans/kmeans_clusters_morphology.png")

logger.info("🎉 KMeans clustering completed successfully.")

# PCA 2D plot
plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=df,
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="Spectral",
    s=70,
    alpha=0.8
)
plt.title(f"KMeans Clusters — PCA 2D Projection (K={best_k})")
plt.suptitle("Clusters visualization in a tight 2D space", y=0.98, fontsize=10, color="gray")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

save_train(plt, f"kmeans/kmeans_pca2d_k{best_k}.png")

logger.info("📸 PCA 2D plot saved.")

# ----------------------------------------------------------
# 🔽 FIXED-K VERSION (OPTION)
# ----------------------------------------------------------
# # 📌 VERSION FIXE — K=4 (whatsoever of the K)

# k_fixed = 4
# kmeans_fixed = KMeans(n_clusters=k_fixed, random_state=42)
# df["Cluster_fixed"] = kmeans_fixed.fit_predict(X_scaled)

# save_model(kmeans_fixed, f"kmeans_fixed_k{k_fixed}.pkl",
#            folder='results/models/kmeans')

# save_clean(df, f"data_with_clusters_k{k_fixed}.csv",
#            folder='viewing/modeling/kmeans')


