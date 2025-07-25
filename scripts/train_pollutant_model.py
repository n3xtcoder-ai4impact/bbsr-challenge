from pollutant_predictor.features.build_features import build_features
from pollutant_predictor.models.train import train_model
from pollutant_predictor.models.inference_pipeline import predict_with_thresholds
from pollutant_predictor.config.paths import PROCESSED_DIR
import pandas as pd

def main():
    import os
    os.makedirs("results", exist_ok=True)

    # Step 1: Build features
    X_train, X_test, y_train, y_test = build_features()

    # Step 2: Train model
    model = train_model(X_train, y_train)

    # Step 3: Evaluate
    thresholds = {"S0": 0.5, "S1": 0.3, "S2": 0.3, "S3": 0.3, "S4": 0.2}
    binary_preds, proba_df, report_df = predict_with_thresholds(model, X_test, y_test, thresholds)

    print("📊 Evaluation Report:")
    print(report_df[["precision", "recall", "f1-score"]])

    # Optional: export results
    report_df.to_csv("results/classification_report.csv")
    proba_df.to_csv("results/test_probabilities.csv", index=False)
    binary_preds.to_csv("results/test_predictions.csv", index=False)

if __name__ == "__main__":
    main()

