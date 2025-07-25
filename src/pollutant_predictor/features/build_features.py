import pandas as pd
from sklearn.model_selection import train_test_split
from pollutant_predictor.data.make_dataset import load_datasets
from pollutant_predictor.config.paths import PROCESSED_DIR

def build_features(test_size=0.2, random_state=42):
    # Load data
    obd_with_pollutants = load_datasets(PROCESSED_DIR / "pollutant_labeled_obd_translated.csv")
    materials_from_components = load_datasets(PROCESSED_DIR / "all_uuid_materials_from_components.csv")
    tbs_df = load_datasets(PROCESSED_DIR / "tbs_deduped.csv")

    # Create combined material context text
    obd_with_pollutants["combined_text"] = (
        obd_with_pollutants["Name (de)"].fillna("") + " " +
        obd_with_pollutants["Kategorie (original)"].fillna("") + " " +
        obd_with_pollutants["productName"].fillna("") + " " +
        obd_with_pollutants["eolCategoryName"].fillna("")
    ).str.lower()

    # Role inference
    role_keywords = {
        "adhesive": ["kleber", "klebstoff", "spachtel"],
        "sealant": ["abdichtung", "dicht", "fuge", "bitumen", "bitumenbahn", "epdm", "eva", "ecb",
                    "pvc", "dachbahn", "unterspannbahn", "kunststoffbahn", "dampfbremse", "folie", "vlies"],
        "mortar": ["mörtel", "zement", "putz", "verputz", "fugenmörtel", "kalkzementputz", "leichtputz", "ausgleichsmasse",
                   "ziegel", "planstein", "leichtbeton", "dachstein", "glasbaustein"],
        "coating": ["farbe", "beschichtung", "lack", "bodenbelag", "linoleum", "korklinoleum", "gussasphaltestrich", "pvc-bodenbelag"],
        "insulation": ["dämm", "wolle", "schaum", "isolierung"],
        "board": ["platte", "gipskarton", "holzfaser"],
        "aggregate": ["kies", "schotter", "sand", "zuschlag", "granulat", "blähton", "naturbims"],
        "metal": ["stahl", "metall", "blech"],
        "wood": ["holz", "sperrholz"]
    }

    def infer_role(text):
        if pd.isna(text):
            return None
        for role, keywords in role_keywords.items():
            if any(keyword in text for keyword in keywords):
                return role
        return "other"

    obd_with_pollutants["material_role"] = obd_with_pollutants["combined_text"].apply(infer_role)

    def refine_sealant_role(row):
        if row["material_role"] != "sealant":
            return row["material_role"]
        text = row["combined_text"]
        if any(x in text for x in ["dachbahn", "epdm", "bitumen", "ecb", "eva"]):
            return "roofing_sealant"
        elif any(x in text for x in ["dampfbremse", "vlies", "folie", "unterspannbahn"]):
            return "vapor_barrier"
        elif any(x in text for x in ["pvc", "bodenbelag", "belag"]):
            return "flooring_sealant"
        else:
            return "sealant"

    obd_with_pollutants["material_role"] = obd_with_pollutants.apply(refine_sealant_role, axis=1)

    # Multi-label pivot
    obd_with_pollutants["target_class"] = obd_with_pollutants["Störstoffklasse"]
    context_cols = ["UUID", "material_role", "eolCategoryName", "eolScenarioUnbuiltReal", "eolScenarioUnbuiltPotential", "technologyFactor"]

    df_multi = obd_with_pollutants[context_cols + ["target_class"]].dropna()
    df_multi["value"] = 1
    df_pivot = df_multi.pivot_table(index=context_cols, columns="target_class", values="value", fill_value=0).reset_index()

    # Prepare X and y
    label_cols = [col for col in df_pivot.columns if col.startswith("S")]
    X = df_pivot.drop(columns=label_cols)
    y = df_pivot[label_cols]

    # One-hot encoding
    X_encoded = pd.get_dummies(X, columns=["material_role", "eolCategoryName", "eolScenarioUnbuiltReal", "eolScenarioUnbuiltPotential"], drop_first=True)
    X_encoded = X_encoded.drop(columns=["UUID"])

    # Train/test split
    return train_test_split(X_encoded, y, test_size=test_size, random_state=random_state)
