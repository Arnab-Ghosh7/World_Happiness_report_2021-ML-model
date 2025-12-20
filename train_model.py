import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import sklearn
import os
import joblib

# --------------------------------------------------
# Version check (IMPORTANT for deployment)
# --------------------------------------------------
print("Scikit-learn version:", sklearn.__version__)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv("world-happiness-report-2021.csv")

# --------------------------------------------------
# Features and target
# --------------------------------------------------
FEATURES = [
    'Logged GDP per capita',
    'Social support',
    'Healthy life expectancy',
    'Freedom to make life choices',
    'Generosity',
    'Perceptions of corruption'
]

TARGET = 'Ladder score'

# --------------------------------------------------
# Clean data
# --------------------------------------------------
df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]

# --------------------------------------------------
# Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Model (deployment-safe configuration)
# --------------------------------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# --------------------------------------------------
# Train model
# --------------------------------------------------
model.fit(X_train, y_train)

# --------------------------------------------------
# Evaluate
# --------------------------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model trained successfully")
print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# --------------------------------------------------
# Save model WITH feature names (CRITICAL)
# --------------------------------------------------
os.makedirs("models", exist_ok=True)

model_artifact = {
    "model": model,
    "features": FEATURES,
    "sklearn_version": sklearn.__version__
}

model_path = os.path.join("models", "happiness_model.pkl")
joblib.dump(model_artifact, model_path)

print(f"Model saved to: {model_path}")
