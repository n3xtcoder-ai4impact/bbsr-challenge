
import pandas as pd
from pathlib import Path
from typing import List
import chardet

def load_obd(obd_path: Path) -> pd.DataFrame:
    """Load raw OBD CSV and apply minimal cleaning."""
    return pd.read_csv(
        obd_path, 
        encoding="ISO-8859-1", 
        delimiter=";", 
        low_memory=False) 
    # e.g. drop duplicates, cast types
    # df = df.drop_duplicates().reset_index(drop=True)

def load_all_obds(obd_dir: Path, pattern: str = "OBD_*_I.csv") -> pd.DataFrame:
    """
    Glob all OBD files matching the pattern in obd_dir,
    load them, and concatenate into one DataFrame.
    """
    files: List[Path] = sorted(obd_dir.glob(pattern))
    dfs = [load_obd(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


# # detect files encoding for pollutant_combinations.csv
# with open("/Users/pablosoriano/Documents/Data Science/bbsr-challenge/data/raw/pollutant_combinations.csv", "rb") as f:
#     result = chardet.detect(f.read(100000))  # Read first 100,000 bytes
#     print(result)

def load_datasets(label_path: Path) -> pd.DataFrame:
    """Load pollutant, tbs and new datasets  CSV."""
    return pd.read_csv(label_path,encoding="utf-8", delimiter=",")


def merge_datasets(obd: pd.DataFrame,
                   labels: pd.DataFrame,
                   on: str = "material_uuid") -> pd.DataFrame:
    """Join OBD data with pollutant labels."""
    merged = obd.merge(labels, on=on, how="left")
    return merged

def save_interim(df: pd.DataFrame, out_path: Path) -> None:
    """Persist cleaned/interim dataset for downstream steps."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

# At bottom of make_dataset.py
if __name__ == "__main__":
    obd = load_obd(Path("data/raw/OBD_2024_I.csv"))
    labels = load_pollutant_labels(Path("data/raw/pollutant_labels.csv"))
    merged = merge_datasets(obd, labels)
    save_interim(merged, Path("data/interim/obd_with_labels.parquet"))
    print("✅ Data pipeline works! Rows:", len(merged))
