import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt

# ==================================================
# LOAD DATASET
# ==================================================

print("\nLoading dataset...")

df = pd.read_csv("perfume_longevity_dataset.csv")

print("\nDataset Loaded Successfully!")
print(df.head())

# ==================================================
# BASIC INFORMATION
# ==================================================

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ==================================================
# FEATURES AND TARGET
# ==================================================

X = df[[
    "Citrus",
    "Floral",
    "Woody",
    "Amber",
    "Musk"
]]

y = df["Longevity"]

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==================================================
# MODEL TRAINING
# ==================================================

print("\nTraining Model...")

model = LinearRegression()

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# ==================================================
# PREDICTIONS
# ==================================================

y_pred = model.predict(X_test)

# ==================================================
# EVALUATION
# ==================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print(f"MAE  : {mae:.2f}")

print(f"MSE  : {mse:.2f}")

print(f"RMSE : {rmse:.2f}")

print(f"R² Score : {r2:.4f}")

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

print("\n========== FEATURE COEFFICIENTS ==========")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coefficients)

# ==================================================
# SAVE MODEL
# ==================================================

os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/perfume_model.pkl"
)

print("\nModel Saved Successfully!")

# ==================================================
# SAMPLE PREDICTIONS
# ==================================================

print("\n========== SAMPLE PREDICTIONS ==========")

sample_perfumes = [

    [4, 2, 0, 0, 0],  # citrus heavy

    [2, 2, 2, 2, 2],  # balanced

    [0, 1, 4, 4, 5]   # strong base notes
]

for perfume in sample_perfumes:

    prediction = model.predict([perfume])[0]

    print(
        f"Input {perfume} -> "
        f"{prediction:.2f} Hours"
    )

# ==================================================
# ACTUAL VS PREDICTED GRAPH
# ==================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Longevity")

plt.ylabel("Predicted Longevity")

plt.title("Actual vs Predicted Perfume Longevity")

plt.grid(True)

plt.savefig(
    "actual_vs_predicted.png",
    bbox_inches="tight"
)

print("\nGraph Saved : actual_vs_predicted.png")

# ==================================================
# RESIDUAL PLOT
# ==================================================

residuals = y_test - y_pred

plt.figure(figsize=(8,6))

plt.scatter(
    y_pred,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Values")

plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.grid(True)

plt.savefig(
    "residual_plot.png",
    bbox_inches="tight"
)

print("Graph Saved : residual_plot.png")

# ==================================================
# END
# ==================================================

print("\nProject Training Completed Successfully!")