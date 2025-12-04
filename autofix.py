# ----------------------------------------------------------
# 📦 IMPORT
# ----------------------------------------------------------
import pandas as pd

from utils.config_utils import AUTOFIX_INPUT, AUTOFIX_OUTPUT

# INPUT_CSV = "data/raw/animal_data_dirty.csv"
# OUTPUT_CSV = INPUT_CSV.replace(".csv", "_reworked.csv")


def clean_csv(path_in=AUTOFIX_INPUT, path_out=AUTOFIX_OUTPUT):
    print(f"📂 Reading: {path_in}")

    try:
        df: pd.DataFrame = pd.read_csv(path_in, sep=";")
    except Exception as e:
        print(f"❌ Reading error: {e}")
        return

    print("✨ CSV successfully loaded using separator ';'")

    # Nettoyage noms de colonnes
    df.columns = (
        df.columns.str.strip()
                 .str.replace(" ", "_", regex=False)
                 .str.replace("-", "_", regex=False)
    )
    print("✨ Column names cleaned")


    df.to_csv(path_out, sep=";", index=False, encoding="utf-8")
    print(f"💾 Clean CSV saved → {path_out}")

if __name__ == "__main__":
    # clean_csv(INPUT_CSV, OUTPUT_CSV)
    clean_csv()