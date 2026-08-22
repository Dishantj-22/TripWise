from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Resolve dataset/model directories relative to this file so the script
# works no matter where it's run from.
DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets"
MODEL_OUT_DIR = Path(__file__).resolve().parent.parent / "models"


df = pd.read_excel(DATASET_DIR / "AI_Travel_Dataset_1000_Records.xlsx")
X = df[
    [
        "Source",
        "Destination",
        "Distance_km",
        "Travelers",
        "Days",
        "VehicleType"
    ]
]

y = df["FinalTripCost"]

categorical = [
    "Source",
    "Destination",
    "VehicleType"
]

numeric = [
    "Distance_km",
    "Travelers",
    "Days"
]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numeric)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

model.fit(X, y)

joblib.dump(model, MODEL_OUT_DIR / "expense_model.pkl")