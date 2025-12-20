import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_PATH = "world-happiness-report-2021.csv"
MODEL_PATH = os.path.join("models", "happiness_model.pkl")
REPORT_PATH = "model_value_report.md"
PLOT_PATH = "feature_importance.png"

# --------------------------------------------------
# Generate Model Report
# --------------------------------------------------
def generate_report():

    # Load data
    df = pd.read_csv(DATA_PATH)

    TARGET = "Ladder score"

    # Load model artifact
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    FEATURES = artifact["features"]

    # Clean data
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    # Train-test split (for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # --------------------------------------------------
    # Feature Importance
    # --------------------------------------------------
    importances = model.feature_importances_

    feature_importance_df = (
        pd.DataFrame({
            "Feature": FEATURES,
            "Importance": importances
        })
        .sort_values(by="Importance", ascending=False)
    )

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_importance_df,
        palette="Blues_r"
    )
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    # --------------------------------------------------
    # Generate Markdown Report
    # --------------------------------------------------
    report_content = f"""# 📊 Model Value Report

## 🔍 Model Overview
- **Model Type:** Random Forest Regressor
- **Dataset:** World Happiness Report 2021

## 📈 Performance Metrics
- **Mean Squared Error (MSE):** {mse:.4f}
- **R-squared (R²):** {r2:.4f}

## ⭐ Feature Importance
The following features were identified as the most influential in predicting
the happiness (ladder) score:

| Feature | Importance |
| :--- | :--- |
"""

    for _, row in feature_importance_df.iterrows():
        report_content += f"| {row['Feature']} | {row['Importance']:.4f} |\n"

    report_content += f"""
![Feature Importance]({PLOT_PATH})
"""

    # Save report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"✅ Report generated successfully at: {REPORT_PATH}")
    print(f"📊 Feature importance plot saved at: {PLOT_PATH}")


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    generate_report()
