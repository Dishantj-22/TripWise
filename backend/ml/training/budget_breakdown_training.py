from pathlib import Path
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Resolve dataset/model directories relative to this file so the script
# works no matter where it's run from.
DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets"
MODEL_OUT_DIR = Path(__file__).resolve().parent.parent / "models"

# ===================================
# Load Dataset
# ===================================

df = pd.read_excel(DATASET_DIR / "Budget_Breakdown_Dataset_1000_Records.xlsx")

# ===================================
# Encode Categorical Features
# ===================================

categorical_columns = [
    "Destination",
    "Trip_Type",
    "Vehicle_Type",
    "Budget_Level",
    "Month"
]

label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ===================================
# Features
# ===================================

X = df[
    [
        "Destination",
        "Days",
        "Travellers",
        "Trip_Type",
        "Vehicle_Type",
        "Budget_Level",
        "Month"
    ]
]

# ===================================
# Targets
# ===================================

y = df[
    [
        "Hotel_Cost",
        "Fuel_Cost",
        "Food_Cost",
        "Shopping_Cost",
        "Misc_Cost"
    ]
]

# ===================================
# Train/Test Split
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===================================
# Train Model
# ===================================

model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
)

model.fit(X_train, y_train)

# ===================================
# Predictions
# ===================================

predictions = model.predict(X_test)

print("\nModel Evaluation")

for i, column in enumerate(y.columns):

    mae = mean_absolute_error(
        y_test.iloc[:, i],
        predictions[:, i]
    )

    r2 = r2_score(
        y_test.iloc[:, i],
        predictions[:, i]
    )

    print(f"{column}")
    print(f"MAE : {mae:.2f}")
    print(f"R²  : {r2:.3f}")
    print()

# ===================================
# Save Model
# ===================================

pickle.dump(
    model,
    open("budget_breakdown_model.pkl", "wb")
)

pickle.dump(
    label_encoders,
    open("budget_breakdown_label_encoders.pkl", "wb")
)

print("Model Saved Successfully!")