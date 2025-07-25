
import pandas as pd
from pathlib import Path

def load_obd(obd_path: Path) -> pd.DataFrame:
    """Load raw OBD CSV and apply minimal cleaning."""
    df = pd.read_csv(obd_path)
    # e.g. drop duplicates, cast types
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def load_pollutant_labels(label_path: Path) -> pd.DataFrame:
    """Load pollutant labels CSV."""
    return pd.read_csv(label_path)

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
