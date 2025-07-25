import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load files
report_df = pd.read_csv("results/classification_report.csv", index_col=0)
preds = pd.read_csv("results/test_predictions.csv")
probs = pd.read_csv("results/test_probabilities.csv")

# Ensure output directory
os.makedirs("results/figures", exist_ok=True)

# --- 1. Per-Class Evaluation Metrics ---
class_df = report_df.loc[["S0", "S1", "S2", "S3", "S4"], ["precision", "recall", "f1-score"]].round(2)
plt.figure(figsize=(8, 5))
sns.barplot(data=class_df.reset_index().melt(id_vars="index"), x="index", y="value", hue="variable", palette="muted")
plt.title("Per-Class Evaluation Metrics")
plt.ylabel("Score")
plt.xlabel("Pollutant Class")
plt.ylim(0, 1.05)
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig("results/figures/performance_barplot.png")
plt.close()

# --- 2. Label Distribution ---
label_counts = preds.sum().sort_values(ascending=False)
plt.figure(figsize=(6, 4))
sns.barplot(x=label_counts.index, y=label_counts.values, hue=label_counts.index, palette="deep", legend=False)
plt.title("Predicted Label Distribution")
plt.ylabel("Number of Samples")
plt.xlabel("Pollutant Class")
plt.tight_layout()
plt.savefig("results/figures/label_distribution.png")
plt.close()

# --- 3. Sample Probabilities with Top Classes ---
# --- 3. Sample Probabilities with Top Classes ---
probs["top_classes"] = probs[["S0", "S1", "S2", "S3", "S4"]].apply(
    lambda row: ", ".join([cls for cls, val in row.items() if val >= 0.3]), axis=1
)
cols = ["top_classes"] + [col for col in probs.columns if col in ["S0", "S1", "S2", "S3", "S4"]]
probs[cols].head(10).to_csv("results/figures/sample_predictions.csv", index=False)

