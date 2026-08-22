from pathlib import Path
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Resolve dataset/model directories relative to this file so the script
# works no matter where it's run from.
DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets"
MODEL_OUT_DIR = Path(__file__).resolve().parent.parent / "models"

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv(DATASET_DIR / "TripWise_Mileage_Dataset_Updated.csv")

# ==========================
# Encode Categorical Columns
# ==========================

categorical_columns = [
    "Brand",
    "Model",
    "Vehicle_Year",
    "Vehicle_Type",
    "Fuel_Type",
    "Transmission",
    "Vehicle_Condition"
]

label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ==========================
# Features & Target
# ==========================

X = df[
    [
        "Brand",
        "Model",
        "Vehicle_Year",
        "Vehicle_Type",
        "Fuel_Type",
        "Transmission",
        "Seating_Capacity",
        "Odometer_km",
        "Vehicle_Condition"
    ]
]

y = df["Mileage"]

# ==========================
# Train/Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train Model
# ==========================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Predictions
# ==========================

predictions = model.predict(X_test)

# ==========================
# Evaluation
# ==========================

print("\nModel Performance")
print("---------------------------")
print("MAE :", round(mean_absolute_error(y_test, predictions), 2))
print("R²  :", round(r2_score(y_test, predictions), 3))

# ==========================
# Save Model
# ==========================

pickle.dump(model, open(MODEL_OUT_DIR / "mileage_model.pkl", "wb"))
pickle.dump(label_encoders, open(MODEL_OUT_DIR / "label_encoders.pkl", "wb"))

print("\nModel Saved Successfully!")
print("Files Created:")
print("mileage_model.pkl")
print("label_encoders.pkl")