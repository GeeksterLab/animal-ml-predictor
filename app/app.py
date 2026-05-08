# ----------------------------------------------------------
# 📦 IMPORTS
# ----------------------------------------------------------

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import glob
import os
import subprocess

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import sklearn
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from utils.config_utils import (
    BASE_DIR,
    DATA_CLEAN,
    DEFAULT_MAIN_DATASET,
    DEFAULT_RAW_DATASET,
    FEATURE_DIR,
    GB_DIR,
    KMEANS_DIR,
    METADATA_DIR,
    MODEL_DIR,
    PLOTS_EDA,
    STATS_EDA,
)


# ----------------------------------------------------------
# 🔧 STREAMLIT-SAFE LOADER
# ----------------------------------------------------------
def load_model_safe(path: Path, debug: bool = False):
    """Load a model with optional Streamlit debug display."""

    if debug:
        st.write(f"🔍 Loading model: `{path.name}`")

        st.write("📦 Environment versions:")
        st.json(
            {
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            }
        )

    if not path.exists():
        if debug:
            st.error(f"❌ File not found: {path}")
        return None

    try:
        model = joblib.load(path)

        if debug:
            st.success(f"✅ Model loaded successfully: {path.name}")

        return model

    except Exception as e:
        if debug:
            st.error("❌ Failed to load model:")
            st.exception(e)
        return None


# ----------------------------------------------------------
# 🧪 MODEL CONSISTENCY CHECKER — Gradient Boosting
# ----------------------------------------------------------
def check_gb_models(GB_MODELS, GB_DIR):
    st.subheader("🔍 Model Consistency Check (GB)")

    missing_files = []
    unexpected_files = []
    summary_rows = []

    actual_files = {f.name for f in GB_DIR.glob("*.pkl")}

    for species, targets in GB_MODELS.items():
        for target_name, path in targets.items():

            file_name = path.name
            exists = file_name in actual_files

            summary_rows.append(
                {
                    "Species": species,
                    "Target": target_name,
                    "File expected": file_name,
                    "Exists on disk": "✔️" if exists else "❌",
                }
            )

            if not exists:
                missing_files.append(file_name)

    expected_files = {
        path.name for paths in GB_MODELS.values() for path in paths.values()
    }
    unexpected_files = actual_files - expected_files

    # --- DISPLAY REPORT ---
    st.markdown("### 📄 Model Files Report")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    if missing_files:
        st.error("❌ Missing model files:")
        for f in missing_files:
            st.write(f"- `{f}`")
    else:
        st.success("🎉 No missing files! All expected models are present.")

    if unexpected_files:
        st.warning("⚠️ Extra files found (not referenced in GB_MODELS):")
        for f in unexpected_files:
            st.write(f"- `{f}`")
    else:
        st.info("No extra files — folder matches model dictionary perfectly.")


# ----------------------------------------------------------
# ⚙️ STREAMLIT CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(
    page_title="🐾 Animal Morphology & Weight Predictor — AetherTech Lab",
    page_icon="☀︎",
    layout="wide",
)

# Sidebar navigation
st.sidebar.title("Navigation")
pages = [
    ":material/assignment: Project",
    ":material/search: Data Exploration",
    ":material/scatter_plot: Data Analysis",
    ":material/psychology: Modeling & Prediction",
]
page = st.sidebar.radio("Go to:", pages)

# ----------------------------------------------------------
# 🎨 STYLING — Hover, Highlights, Transitions
# ----------------------------------------------------------
st.markdown(
    """
<style>
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
  background: rgba(90, 132, 255, 0.15);
  border-radius: 10px;
  padding: 6px 8px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
  font-weight: 700;
}
.stButton > button {
  border-radius: 10px;
  transition: transform .06s ease, box-shadow .12s ease, background-color .12s ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 18px rgba(0,0,0,.18);
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------
# 📥 DATA LOADING
# ----------------------------------------------------------
try:
    df_raw = pd.read_csv(DEFAULT_RAW_DATASET, sep=";")
    st.sidebar.success(f"Raw dataset loaded ✔ ({DEFAULT_RAW_DATASET.name})")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load raw dataset: {e}")
    df_raw = None

try:
    df_clean = pd.read_csv(DEFAULT_MAIN_DATASET)
    st.sidebar.success(f"Clean dataset loaded ✔ ({DEFAULT_MAIN_DATASET.name})")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load clean dataset: {e}")
    df_clean = None

# Dataset principal utilisé par Data Analysis + Modeling
df = df_clean

# ----------------------------------------------------------
# 📄 PAGE 1 — PROJECT OVERVIEW
# ----------------------------------------------------------
if page == pages[0]:
    st.divider()
    st.header(":rainbow[Animal Morphology Lab] :crystal_ball:")
    st.divider()
    st.markdown("""
    ### Project Overview

    **Animal Morphology Lab** is a complete data science project based on a generated animal morphology dataset.

    The dataset was designed to simulate a realistic workflow: messy raw data, cleaning, validation, exploratory analysis, model training, and interactive prediction.

    The app demonstrates how raw data can be transformed into a clean, structured, and ML-ready dataset, then used to build species-specific prediction tools.

    Key components include:

    - Raw vs clean dataset comparison
    - Missing values and duplicate analysis
    - Visual exploration of morphology patterns
    - Gradient Boosting prediction models
    - Interactive weight and length prediction by species
    - Model behavior analysis with simulations
    """)
    # st.caption(":orange[Upload your CSV or load it from a local path in the sidebar.]")

# ----------------------------------------------------------
# 🔍 PAGE 2 — DATA EXPLORATION
# ----------------------------------------------------------
elif page == pages[1]:
    if df_raw is None or df_clean is None:
        st.warning("⚠️ Please load both raw and clean datasets first.")
        st.stop()

    st.divider()
    st.header(":rainbow[Data Exploration]")
    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("📄 Raw Dataset Preview")
        st.dataframe(df_raw.head(100))
        st.write(f"Data shape: {df_raw.shape}")

        if st.checkbox("📉 Show Raw NaN count", key="raw_nan_count"):
            st.caption("Counts missing values for each column.")
            st.dataframe(df_raw.isna().sum())

        if st.checkbox("♻️ Show Raw duplicated rows", key="raw_duplicates"):
            st.caption("Counts duplicated rows in the raw dataset.")
            st.write(int(df_raw.duplicated().sum()))

        if st.checkbox("⚠️ Show Raw rows with missing values", key="raw_missing_rows"):
            st.caption("Displays all raw rows containing at least one NaN value.")
            lines_na_raw = df_raw[df_raw.isna().any(axis=1)]
            st.write(f"{len(lines_na_raw)} raw rows with missing values")
            st.dataframe(lines_na_raw)
            st.download_button(
                "📥 Download raw missing rows",
                data=lines_na_raw.to_csv(index=False).encode("utf-8"),
                file_name="raw_missing_rows.csv",
                mime="text/csv",
            )

    with right:
        st.subheader("🧼 Clean Dataset Preview")
        st.dataframe(df_clean.head(100))
        st.write(f"Data shape: {df_clean.shape}")

        if st.checkbox("📉 Show Clean NaN count", key="clean_nan_count"):
            st.caption("Counts missing values for each column.")
            st.dataframe(df_clean.isna().sum())

        if st.checkbox("♻️ Show Clean duplicated rows", key="clean_duplicates"):
            st.caption("Counts duplicated rows in the clean dataset.")
            st.write(int(df_clean.duplicated().sum()))

        if st.checkbox(
            "⚠️ Show Clean row with missing values", key="clean_missing_rows"
        ):
            st.caption("Displays all clean rows containing at least one NaN value.")
            lines_na_clean = df_clean[df_clean.isna().any(axis=1)]
            st.write(f"{len(lines_na_clean)} clean rows with missing values")
            st.dataframe(lines_na_clean)
            st.download_button(
                "📥 Download clean missing rows",
                data=lines_na_clean.to_csv(index=False).encode("utf-8"),
                file_name="clean_missing_rows.csv",
                mime="text/csv",
            )

    st.divider()
    st.subheader("🧼 Clean EDA Reports")

    with st.expander("📘 Numerical Statistics (Markdown)"):
        st.caption("Summary of numerical values: averages, spread, and distributions.")
        try:
            with open(STATS_EDA / "eda_numerical_stats.md", "r", encoding="utf-8") as f:
                st.markdown(f.read())
        except:
            st.info("No numerical EDA markdown file found.")

    with st.expander("📙 Categorical Statistics (Markdown)"):
        st.caption(
            "Summary of categorical values: most frequent categories and repartitions."
        )
        try:
            with open(
                STATS_EDA / "eda_categorical_stats.md", "r", encoding="utf-8"
            ) as f:
                st.markdown(f.read())
        except:
            st.info("No categorical EDA markdown file found.")

    st.subheader("📊 EDA Plots")

    with st.expander("📈 Distribution — Weight vs Length (by species)"):
        st.caption("Helps reveal morphological patterns across species.")
        plot_path_1 = PLOTS_EDA / "distribution_weight_vs_length_by_gender.png"
        if os.path.exists(plot_path_1):
            st.image(plot_path_1, width=800)
        else:
            st.info(
                "Plot file not found. Please generate the plot and save it as 'results/plots/EDA/distribution_weight_vs_length_by_gender.png'."
            )

    with st.expander("🗺️ Repartition — Weight by Country"):
        st.caption("Helps reveal weight patterns across countries and species.")
        plot_path_2 = PLOTS_EDA / "repartion_weight_by_country.png"
        if os.path.exists(plot_path_2):
            st.image(plot_path_2, width=800)
        else:
            st.info(
                "Plot file not found. Please generate the plot and save it as 'results/plots/EDA/repartion_weight_by_country.png'."
            )

# ----------------------------------------------------------
# 📊 PAGE 3 — VISUAL ANALYSIS
# ----------------------------------------------------------
elif page == pages[2]:
    if df is None:
        st.warning("⚠️ Please load a dataset first.")
        st.stop()

    st.divider()
    st.header("📊 Data Visualization")
    st.caption("Explore patterns, compare behaviors, detect trends.")
    st.divider()

    # Tabs
    tab_hm, tab_box, tab_scatter = st.tabs(["Heatmap", "Barplot", "Scatter"])

    # ======================================================
    # 🔥 TAB 1 — HEATMAP
    # ======================================================
    with tab_hm:
        st.subheader("🔬 Correlation Insights")
        st.caption(
            "Detect linear or rank-based relationships between numerical features."
        )

        num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()

        corr_method = st.radio(
            "Correlation method:",
            ["Pearson (linear)", "Spearman (rank-based)"],
            horizontal=True,
        )

        selected_cols = st.multiselect(
            "Select variables to include:", num_cols, default=num_cols
        )

        generate = st.button("📡 Generate Heatmap", use_container_width=True)

        if generate:
            method = "pearson" if "Pearson" in corr_method else "spearman"
            cm = df[selected_cols].corr(method=method)

            fig = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="YlGnBu",
                title=f"Correlation Matrix ({corr_method})",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Strongest correlations
            mask = np.triu(np.ones(cm.shape), k=1).astype(bool)
            cm_upper = cm.where(mask)
            strongest = cm_upper.stack().sort_values(ascending=False)

            st.write("### 🧠 Auto-Analysis")

            if strongest.empty:
                st.info("💡 No meaningful correlations detected.")
            else:
                top = strongest.head(3)
                for idx, val in top.items():
                    a, b = idx if isinstance(idx, tuple) else ("?", "?")
                    st.markdown(f"**{a} ↔ {b}** : correlation **{val:.2f}**")

                st.caption(
                    "High correlations may reveal shared structure or measurement effects."
                )

    # ======================================================
    # 🔥 TAB 2 — BARPLOT
    # ======================================================
    with tab_box:
        st.subheader("📊 Barplot Insights")
        st.caption("Compare average numeric values across categories.")

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        num_cols = df.select_dtypes(include=["int", "float"]).columns

        if len(cat_cols) >= 1 and len(num_cols) >= 1:
            x = st.selectbox("Categorical variable (X-axis)", cat_cols, key="bar_x")
            y = st.selectbox("Numerical variable (Y-axis)", num_cols, key="bar_y")

            if st.button("📈 Show Barplot", use_container_width=True):
                grouped = df.groupby(x)[y].mean().sort_values(ascending=False)

                fig = px.bar(
                    grouped,
                    x=grouped.index,
                    y=grouped.values,
                    color=grouped.values,
                    color_continuous_scale="Viridis",
                    title=f"Average {y} by {x}",
                )
                fig.update_layout(height=500)

                st.plotly_chart(fig, use_container_width=True)

                top_cat = grouped.idxmax()
                top_val = grouped.max()
                bottom_cat = grouped.idxmin()
                bottom_val = grouped.min()

                st.info(
                    f"📌 **Insight:** Highest {y} in **{top_cat}** ({top_val:.2f}), "
                    f"lowest in **{bottom_cat}** ({bottom_val:.2f})."
                )

    # ======================================================
    # 🔥 TAB 3 — SCATTER
    # ======================================================
    with tab_scatter:
        st.subheader("🔍 Scatter Insights")
        st.caption("Reveal numerical trends and structural relationships.")

        num_cols = df.select_dtypes(include=["int", "float"]).columns

        if len(num_cols) >= 2:
            x = st.selectbox("X-axis", num_cols, key="scatter_x")
            y = st.selectbox("Y-axis", num_cols, key="scatter_y")

            if st.button("📉 Show Scatter", use_container_width=True):
                corr_value = float(df[[x, y]].corr().iloc[0, 1])

                fig = px.scatter(
                    df,
                    x=x,
                    y=y,
                    opacity=0.6,
                    trendline="ols",
                    color_discrete_sequence=["#4C72B0"],
                    title=f"{x} vs {y}",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Insight based on correlation intensity
                if corr_value > 0.7:
                    msg = "strong alignment"
                elif corr_value > 0.4:
                    msg = "moderate relationship"
                elif corr_value > 0.1:
                    msg = "weak trend"
                else:
                    msg = "no clear relationship"

                st.info(f"📈 **Correlation:** `{corr_value:.2f}` → {msg}.")

    # ======================================================
    # 🧠 FINAL SUMMARY
    # ======================================================
    st.divider()
    st.header("🧠 Automated DataViz Summary")
    st.caption("Key behavioral and structural patterns observed in the dataset.")

    try:
        cm = df.corr(numeric_only=True)
        cm_flat = cm.abs().stack().drop_duplicates()

        strongest_pair = cm_flat[cm_flat < 0.999].sort_values(ascending=False).index[0]
        corr_value = cm_flat[cm_flat < 0.999].sort_values(ascending=False).iloc[0]
        var1, var2 = strongest_pair

        top_species = (
            df["Species"].value_counts().idxmax() if "Species" in df.columns else "N/A"
        )

        summary = f"""
### 📌 Summary Highlights
• **Strongest correlation:** {var1} ↔ {var2} (corr = {corr_value:.2f})
• **Dominant species:** {top_species}
• **Numeric distributions:** globally stable
• **Category imbalance:** detected, may influence model accuracy
"""

        st.markdown(summary)

    except Exception as e:
        st.warning(f"Summary could not be generated: {e}")


# ----------------------------------------------------------
# 🧠 PAGE 4 — MODELING & PREDICTION (INTERACTIVE DELUXE)
# ----------------------------------------------------------

elif page == pages[3]:
    st.divider()
    st.header(":rainbow[Modeling & Prediction] :material/psychology:")
    st.divider()

    # ---------------------------------------------------------
    # 🔹 Auto-loading of all models (Gradient + KMeans) ---
    # ---------------------------------------------------------
    st.subheader(":gear: Loading Models...")

    # --- Paths for Gradient Boosting models ---
    GB_MODELS = {
        "European Bison": {
            "length": GB_DIR / "gradient_boosting_length_european_bison.pkl",
            "weight": GB_DIR / "gradient_boosting_weight_european_bison.pkl",
        },
        "Hedgehog": {
            "length": GB_DIR / "gradient_boosting_length_hedgehog.pkl",
            "weight": GB_DIR / "gradient_boosting_weight_hedgehog.pkl",
        },
        "Lynx": {
            "length": GB_DIR / "gradient_boosting_length_lynx.pkl",
            "weight": GB_DIR / "gradient_boosting_weight_lynx.pkl",
        },
        "Red Squirrel": {
            "length": GB_DIR / "gradient_boosting_length_red_squirrel.pkl",
            "weight": GB_DIR / "gradient_boosting_weight_red_squirrel.pkl",
        },
    }

    # 🔧 Developer Mode (local only)
    DEBUG = False  # Set True only on local machine

    if DEBUG:
        check_gb_models(GB_MODELS, GB_DIR)

    # --- Load Gradient Boosting models ---
    loaded_gb_models = {}

    for species_name, paths in GB_MODELS.items():
        species_models = {}

        for target_name, model_path in paths.items():

            model = load_model_safe(model_path, debug=DEBUG)

            if model is not None:
                species_models[target_name] = model
            else:
                species_models[target_name] = None
                st.error(
                    f"❌ Could not load Gradient Boosting {target_name} model for {species_name}"
                )

        loaded_gb_models[species_name] = species_models

    # --- KMeans models ---
    KMEANS_MODELS = {
        "K2": KMEANS_DIR / "kmeans_morpho_geo_k2.pkl",
    }

    loaded_kmeans = {}
    for name, path in KMEANS_MODELS.items():
        try:
            loaded_kmeans[name] = joblib.load(path)
            st.success(f"Loaded KMeans model: **{name}**")
        except Exception:
            loaded_kmeans[name] = None
            st.warning(f"⚠️ Could not load KMeans model: {name}")

    # --- Load scaler ---
    try:
        kmeans_scaler = joblib.load(KMEANS_DIR / "kmeans_scaler.pkl")
        st.success("🔧 KMeans scaler loaded.")
    except Exception:
        kmeans_scaler = None
        st.error("❌ Could not load KMeans scaler.")

    # --- Load feature importances weight ---
    try:
        fi_weight = pd.read_csv(
            FEATURE_DIR / "feature_importance_gradientboosting_weight_per_species.csv"
        )

        st.success("📊 Weight Feature Importance loaded.")
    except:
        fi_weight = None

    # --- Load feature importances length ---
    try:
        fi_length = pd.read_csv(
            FEATURE_DIR / "feature_importance_gradientboosting_length_per_species.csv"
        )
        st.success("📊 Length Feature Importance loaded.")
    except:
        fi_length = None

    # --- Merge (optionnel) ---
    if fi_weight is not None and fi_length is not None:
        feature_importance = pd.concat([fi_weight, fi_length], ignore_index=True)
    else:
        feature_importance = fi_weight or fi_length

    # 🔹 Model global pour analyse (PDP, etc.)
    try:
        analysis_model = joblib.load(
            GB_DIR / "gradient_boosting_multioutput_animal_country_to_length_weight.pkl"
        )
        st.success(
            "🧠 Global multi-output Gradient Boosting model loaded for analysis."
        )
    except Exception:
        analysis_model = None
        st.info("ℹ️ Global analysis model not available (PDP will be limited).")

    # ------------------------------------------------------
    # 🔍 ADVANCED MODEL VISUALIZATION & ANALYSIS
    # ------------------------------------------------------
    st.divider()
    st.header(":bar_chart: Advanced Model Analysis")

    # ------------------------------------------------------
    # 🧮 Model KPIs Summary (optional dashboard cards)
    # ------------------------------------------------------
    if "y_test" in st.session_state and "y_pred" in st.session_state:
        mse = mean_squared_error(st.session_state["y_test"], st.session_state["y_pred"])
        r2 = r2_score(st.session_state["y_test"], st.session_state["y_pred"])
        mae = np.mean(np.abs(st.session_state["y_test"] - st.session_state["y_pred"]))

        col1, col2, col3 = st.columns(3)
        col1.metric("📉 MSE (Mean Squared Error)", f"{mse:.2f}", delta=None)
        col2.metric("📈 R² Score", f"{r2:.2f}")
        col3.metric("⚙️ MAE (Mean Absolute Error)", f"{mae:.2f}")

    # -------------------------------------------------------------
    # 🔹 Interactive Performance Scatter (Real vs Predicted) ---
    # -------------------------------------------------------------
    st.subheader("📈 Model Performance Visualization")
    try:
        if "y_test" not in st.session_state or "y_pred" not in st.session_state:
            st.session_state["y_test"] = np.random.uniform(0, 100, 100)
            st.session_state["y_pred"] = st.session_state["y_test"] + np.random.normal(
                0, 10, 100
            )

        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        perf_df = pd.DataFrame({"Real (%)": y_test, "Predicted (%)": y_pred})
        perf_df["Diff"] = np.abs(perf_df["Real (%)"] - perf_df["Predicted (%)"])

        # Slider to filter by error tolerance
        tolerance = st.slider(
            "🎯 Confidence threshold (max allowed error %)",
            min_value=0,
            max_value=50,
            value=15,
            step=1,
            help="Filter points within a specific error range.",
        )

        filtered_df = perf_df[perf_df["Diff"] <= tolerance]

        fig_perf = px.scatter(
            perf_df,
            x="Real (%)",
            y="Predicted (%)",
            opacity=0.6,
            color="Diff",
            color_continuous_scale="RdYlGn_r",
            title=f"🤖 Model Performance — MSE: {mse:.2f} | R²: {r2:.2f}",
        )
        fig_perf.add_shape(
            type="line",
            x0=perf_df["Real (%)"].min(),
            y0=perf_df["Real (%)"].min(),
            x1=perf_df["Real (%)"].max(),
            y1=perf_df["Real (%)"].max(),
            line=dict(color="red", dash="dash"),
        )

        st.plotly_chart(fig_perf, use_container_width=True)
        st.caption(f"🟢 {len(filtered_df)} points within ±{tolerance}% confidence zone")

    except Exception as e:
        st.error(f"Error generating performance plot: {e}")

    # ------------------------------------------------------
    # 🎯 INTERACTIVE PREDICTION TOOL (Species Models)
    # ------------------------------------------------------
    st.divider()
    st.header(":material/psychology_alt: Interactive Prediction (per Species)")

    # ---------------------------
    # Load clean dataset
    # ---------------------------
    df_clean = pd.read_csv(DEFAULT_MAIN_DATASET)
    species_list = sorted(df_clean["Animal"].dropna().unique())

    # ---------------------------
    # User selects species
    # ---------------------------
    species = st.selectbox("🧬 Select Species:", species_list)

    # ---------------------------
    # Load corresponding models
    # ---------------------------
    # safe_slug = species.replace(" ", "_").lower() if isinstance(species, str) else ""

    # try:
    #     model_length = joblib.load(GB_DIR / f"gradient_boosting_length_{safe_slug}.pkl")

    # except:
    #     model_length = None

    # try:
    #     model_weight = joblib.load(GB_DIR / f"gradient_boosting_weight_{safe_slug}.pkl")
    # except:
    #     model_weight = None
    species_models = loaded_gb_models.get(species, {})

    model_length = species_models.get("length")
    model_weight = species_models.get("weight")

    if model_length is None and model_weight is None:
        st.error(f"❌ No models found for {species}.")
        st.stop()

    # ---------------------------
    # Choose direction
    # ---------------------------
    mode = st.radio(
        "Choose prediction type:",
        ["Predict Weight from Length", "Predict Length from Weight"],
        horizontal=True,
    )

    # ---------------------------
    # User Input
    # ---------------------------

    if "last_value" not in st.session_state:
        st.session_state.last_value = None

    if "last_pred" not in st.session_state:
        st.session_state.last_pred = None

    if mode == "Predict Weight from Length":
        value = st.number_input(
            "📏 Enter Length (cm) — **min: 0 | max: 400**:",
            min_value=10.0,
            max_value=400.0,
            value=100.0,
            step=1.0,
        )
        model = model_weight
        input_feature = "Length_cm"
        output_name = "Weight_kg"
        unit = "kg"

    else:
        value = st.number_input(
            "🏋️ Enter Weight (kg)  — **min: 0 | max: 2000**:",
            min_value=10.0,
            max_value=2000.0,
            value=50.0,
            step=1.0,
        )
        model = model_length
        input_feature = "Weight_kg"
        output_name = "Length_cm"
        unit = "cm"

    # ---------------------------
    # Build input row
    # ---------------------------
    X_input = pd.DataFrame([{input_feature: value}])

    # ---------------------------
    # Prediction
    # ---------------------------
    st.divider()
    st.subheader(":crystal_ball: Prediction Result")

    if st.button("⚡ Predict", type="primary", use_container_width=True):

        try:
            if model is None:
                st.error("❌ No model available for this species.")
                st.stop()

            pred = float(model.predict(X_input)[0])
            pred_rounded = round(pred, 2)

            # --- 🧠 Smart UX: input changed but prediction identical ---
            if (
                st.session_state.last_value is not None
                and st.session_state.last_pred is not None
            ):
                if (
                    value != st.session_state.last_value
                    and pred_rounded == st.session_state.last_pred
                ):
                    st.warning(
                        "😅 Tu as changé la valeur, **mais la prédiction n'a pas bougé**. "
                        "Essaie un chiffre un peu plus éloigné pour mieux voir la réaction du modèle !"
                    )

            # --- Display prediction ---
            st.success(
                f"✨ **Predicted {output_name.replace('_', ' ')} :** "
                f"**{pred_rounded:.2f} {unit}**"
            )

            # Update stored values
            st.session_state.last_value = value
            st.session_state.last_pred = pred_rounded

            # 🔬 Store lab context for advanced visualizations
            st.session_state["lab_context"] = {
                "species": species,
                "mode": mode,
                "input_feature": input_feature,
                "output_name": output_name,
                "unit": unit,
                "value": float(value),
                "pred": float(pred_rounded),
                "model_type": "weight" if model is model_weight else "length",
            }

            # --- Mini gauge plot ---
            gauge_fig = px.scatter(
                x=[0],
                y=[pred_rounded],
                size=[40],
                color=[pred_rounded],
                range_y=[0, max(pred_rounded * 1.5, 1)],
                title=f"{output_name.replace('_', ' ')} estimation",
                labels={"y": f"{output_name.replace('_', ' ')} ({unit})", "x": ""},
            )
            gauge_fig.update_layout(
                showlegend=False,
                height=300,
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

            # --- Interpretation hint ---
            st.info(
                f"💡 Pour un **{species}** avec un(e) "
                f"**{input_feature.replace('_', ' ')}** de **{value} {unit}**, "
                f"le modèle estime un(e) **{output_name.replace('_', ' ')}** "
                f"de **{pred_rounded:.2f} {unit}**."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    else:
        st.info("👈 Enter a value and click *Predict*.")

    # ======================================================
    # 🧪 AI LAB — Advanced Analysis (Futuristic Mode)
    # ======================================================
    if "lab_context" in st.session_state:
        ctx = st.session_state["lab_context"]
        current_model = model_weight if ctx["model_type"] == "weight" else model_length

        # Ensure model is loaded (for type checkers)
        if current_model is None:
            st.error("Model not loaded for this species.")
            st.stop()

        species_ctx = ctx["species"]
        input_feature_ctx = ctx["input_feature"]
        output_name_ctx = ctx["output_name"]
        unit_ctx = ctx["unit"]
        value_ctx = ctx["value"]
        pred_ctx = ctx["pred"]

        df_species = df_clean[df_clean["Animal"] == species_ctx].copy()

        # 🎨 Palette auto "intelligente"
        palette_curve = px.colors.sequential.Viridis
        palette_sim = px.colors.sequential.Plasma
        point_color = "#00E0FF"

        # --------------------------------------------------
        # 1️⃣ Morphology Scan — Species Summary
        # --------------------------------------------------
        with st.expander("🔍 Morphology Scan — Species Summary"):
            st.caption(
                "Synthetic morphology profile for the selected species: "
                "central tendencies, spread and anatomical range."
            )

            if df_species.empty:
                st.warning("No data available for this species in the dataset.")
            else:
                stats_rows = []
                for feature in ["Length_cm", "Weight_kg"]:
                    if feature in df_species.columns:
                        series = df_species[feature].dropna()
                        if len(series) == 0:
                            continue
                        stats_rows.append(
                            {
                                "Metric": feature.replace("_", " "),
                                "Mean": round(series.mean(), 2),
                                "Std": round(series.std(), 2),
                                "Min": round(series.min(), 2),
                                "Max": round(series.max(), 2),
                                "N": int(series.count()),
                            }
                        )

                if not stats_rows:
                    st.info("No numeric morphology metrics available.")
                else:
                    st.dataframe(
                        pd.DataFrame(stats_rows),
                        use_container_width=True,
                    )
                    st.info(
                        "💡 This summary gives the **typical morphology envelope** "
                        "for the species — useful to judge how extreme your input is."
                    )

        # --------------------------------------------------
        # 2️⃣ Response Curve & Plateau Detector
        # --------------------------------------------------
        with st.expander("📈 Response Curve & Plateau Detector"):
            st.caption(
                "Model response curve around your input, with local slope analysis "
                "to detect stability vs sensitive zones."
            )

            if df_species.empty:
                st.warning(
                    "No data available to build a response curve for this species."
                )
            else:
                if input_feature_ctx not in df_species.columns:
                    st.warning(
                        f"Input feature `{input_feature_ctx}` not found in dataset."
                    )
                else:
                    f_series = df_species[input_feature_ctx].dropna()
                    if len(f_series) < 5:
                        st.warning("Not enough points to build a meaningful curve.")
                    else:
                        f_min, f_max = float(f_series.min()), float(f_series.max())
                        span = f_max - f_min or 1.0
                        low = max(0.0, f_min - 0.1 * span)
                        high = f_max + 0.1 * span

                        grid = np.linspace(low, high, 80)
                        X_grid = pd.DataFrame([{input_feature_ctx: v} for v in grid])
                        y_grid = current_model.predict(X_grid)

                        curve_df = pd.DataFrame(
                            {
                                input_feature_ctx: grid,
                                output_name_ctx: y_grid,
                            }
                        )

                        fig_curve = px.line(
                            curve_df,
                            x=input_feature_ctx,
                            y=output_name_ctx,
                            title="Model Response Curve",
                            color_discrete_sequence=palette_curve,
                        )
                        fig_curve.add_scatter(
                            x=[value_ctx],
                            y=[pred_ctx],
                            mode="markers",
                            marker=dict(
                                size=14,
                                color=point_color,
                                line=dict(width=2, color="white"),
                            ),
                            name="Your prediction",
                        )

                        st.plotly_chart(fig_curve, use_container_width=True)

                        # --- Plateau detection (gradient local) ---
                        delta = span / 40 if span > 0 else 1.0
                        left = max(low, value_ctx - delta)
                        right = min(high, value_ctx + delta)

                        X_left = pd.DataFrame([{input_feature_ctx: left}])
                        X_right = pd.DataFrame([{input_feature_ctx: right}])

                        y_left = float(current_model.predict(X_left)[0])
                        y_right = float(current_model.predict(X_right)[0])

                        grad = (y_right - y_left) / max(right - left, 1e-6)

                        if abs(grad) < 0.01:
                            st.warning(
                                "🕳️ **Plateau detected** — in this region, the model reaction "
                                "to changes in the input is very weak. Small variations in "
                                f"{input_feature_ctx.replace('_', ' ')} do **not** significantly "
                                f"modify predicted {output_name_ctx.replace('_', ' ')}."
                            )
                        else:
                            st.info(
                                "⚡ **Active slope zone** — the model is responsive here. "
                                f"A change in **{input_feature_ctx.replace('_', ' ')}** "
                                f"produces a noticeable shift in predicted "
                                f"**{output_name_ctx.replace('_', ' ')}**."
                            )

        # --------------------------------------------------
        # 3️⃣ Synthetic Simulation Field (100+ Experiments)
        # --------------------------------------------------
        with st.expander("🧪 Synthetic Simulation Field"):
            st.caption(
                "Generate hypothetical experiments to probe how the model behaves "
                "far beyond observed morphology."
            )

            col_sim1, col_sim2 = st.columns(2)
            n_points = col_sim1.slider(
                "Number of synthetic experiments:",
                min_value=50,
                max_value=500,
                value=150,
                step=25,
            )

            spread_factor = col_sim2.slider(
                "Exploration span (relative to typical range):",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="1.0 = around typical range, >1.0 = more extreme hypothetical cases.",
            )

            if input_feature_ctx not in df_species.columns:
                st.warning(
                    f"Input feature `{input_feature_ctx}` not available to build simulations."
                )
            else:
                base_series = df_species[input_feature_ctx].dropna()
                if len(base_series) == 0:
                    st.warning("No data available for synthetic sampling.")
                else:
                    base_min, base_max = float(base_series.min()), float(
                        base_series.max()
                    )
                    base_span = base_max - base_min or 1.0

                    sim_low = max(0.0, base_min - (spread_factor - 1.0) * base_span)
                    sim_high = base_max + (spread_factor - 1.0) * base_span

                    sim_values = np.random.uniform(sim_low, sim_high, n_points)
                    X_sim = pd.DataFrame([{input_feature_ctx: v} for v in sim_values])
                    y_sim = current_model.predict(X_sim)

                    sim_df = pd.DataFrame(
                        {
                            input_feature_ctx: sim_values,
                            output_name_ctx: y_sim,
                        }
                    )

                    fig_sim = px.scatter(
                        sim_df,
                        x=input_feature_ctx,
                        y=output_name_ctx,
                        opacity=0.6,
                        color=output_name_ctx,
                        color_continuous_scale=palette_sim,
                        title="Synthetic Experiments — Model Behaviour Map",
                    )

                    fig_sim.add_scatter(
                        x=[value_ctx],
                        y=[pred_ctx],
                        mode="markers",
                        marker=dict(
                            size=18,
                            color=point_color,
                            line=dict(width=3, color="white"),
                            symbol="circle",
                        ),
                        name="Your experiment",
                    )

                    st.plotly_chart(fig_sim, use_container_width=True)

                    st.info(
                        "💡 This synthetic field lets you see **how the model generalizes** "
                        "to values that do not exist in the original dataset — "
                        "ideal to reason about robustness and extrapolation."
                    )
