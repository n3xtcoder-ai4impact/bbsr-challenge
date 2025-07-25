from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

def train_model(X_train, y_train, model_type="random_forest", save_path: Path = None):
    """
    Trains a multi-label classification model and optionally saves it.

    Parameters:
    - X_train: training features
    - y_train: training labels
    - model_type: "random_forest" or "logistic_regression"
    - save_path: optional path to save the trained model (joblib)

    Returns:
    - trained MultiOutputClassifier
    """
    if model_type == "random_forest":
        base_model = RandomForestClassifier(n_estimators=200, random_state=42)
    elif model_type == "logistic_regression":
        base_model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = MultiOutputClassifier(base_model)
    model.fit(X_train, y_train)

    if save_path:
        joblib.dump(model, save_path)
        print(f"✅ Model saved to {save_path}")

    return model


