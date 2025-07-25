import pandas as pd
from sklearn.metrics import classification_report

def predict_with_thresholds(model, X_test, y_test, thresholds: dict):
    """
    Predicts probabilities and applies class-specific thresholds.
    Returns the thresholded predictions and evaluation report.
    """
    y_proba = model.predict_proba(X_test)

    proba_df = pd.DataFrame({
        class_name: probs[:, 1] for class_name, probs in zip(y_test.columns, y_proba)
    })

    binary_predictions = pd.DataFrame({
        class_name: (proba_df[class_name] >= thresholds.get(class_name, 0.5)).astype(int)
        for class_name in y_test.columns
    })

    report = classification_report(y_test, binary_predictions, target_names=y_test.columns, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    return binary_predictions, proba_df, report_df


def predict_unlabeled_tbs(model, tbs_df, X_train_columns, thresholds: dict):
    """
    Takes a trained model and TBS dataframe, performs feature engineering,
    and predicts pollutant class probabilities and binary labels.
    """
    tbs_df = tbs_df.copy()
    tbs_df["combined_text"] = (
        tbs_df["productName"].fillna("") + " " +
        tbs_df["eolCategoryName"].fillna("")
    ).str.lower()

    def infer_role(text):
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
        for role, keywords in role_keywords.items():
            if any(keyword in text for keyword in keywords):
                return role
        return "other"

    def refine_sealant(row):
        if row["material_role"] != "sealant": return row["material_role"]
        text = row["combined_text"]
        if any(x in text for x in ["dachbahn", "epdm", "bitumen", "ecb", "eva"]): return "roofing_sealant"
        if any(x in text for x in ["dampfbremse", "vlies", "folie", "unterspannbahn"]): return "vapor_barrier"
        if any(x in text for x in ["pvc", "bodenbelag", "belag"]): return "flooring_sealant"
        return "sealant"

    tbs_df["material_role"] = tbs_df["combined_text"].apply(infer_role)
    tbs_df["material_role"] = tbs_df.apply(refine_sealant, axis=1)

    context_cols = [
        "material_role", "eolCategoryName", 
        "eolScenarioUnbuiltReal", "eolScenarioUnbuiltPotential", 
        "technologyFactor"
    ]
    context = tbs_df[context_cols].copy()

    # Encode with same dummies as training
    X_encoded = pd.get_dummies(context, drop_first=True)
    X_encoded = X_encoded.reindex(columns=X_train_columns, fill_value=0)

    # Predict probabilities
    proba = model.predict_proba(X_encoded)
    proba_df = pd.DataFrame({
        class_name: probs[:, 1] for class_name, probs in zip(model.classes_, proba)
    })

    # Apply thresholds
    binary_preds = pd.DataFrame({
        class_name: (proba_df[class_name] >= thresholds.get(class_name, 0.5)).astype(int)
        for class_name in proba_df.columns
    })

    return proba_df, binary_preds

def blend_predictions_with_components(pred_df, uuid_map, uuid_col="UUID", component_col="main_component_id"):
    """
    Blends material-level predictions with component-level averages.
    Expects:
    - pred_df: DataFrame with predictions and UUID
    - uuid_map: DataFrame with UUID to component ID mapping
    Returns:
    - blended DataFrame with adjusted probabilities
    """
    df = pred_df.merge(uuid_map[[uuid_col, component_col]], on=uuid_col, how="left")

    # Identify probability columns
    prob_cols = [col for col in df.columns if col.startswith("S") and not col.endswith("_adjusted")]

    # Component-level averages
    comp_avg = df.groupby(component_col)[prob_cols].mean().reset_index()

    # Merge + blend
    blended = df.merge(comp_avg, on=component_col, suffixes=("_mat", "_comp"))
    for col in prob_cols:
        blended[f"{col}_adjusted"] = 0.7 * blended[f"{col}_mat"] + 0.3 * blended[f"{col}_comp"]

    return blended