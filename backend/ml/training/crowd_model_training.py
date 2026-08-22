from pathlib import Path
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Resolve dataset/model directories relative to this file so the script
# works no matter where it's run from.
DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets"
MODEL_OUT_DIR = Path(__file__).resolve().parent.parent / "models"

# ==========================
# Load Dataset
# ==========================

df = pd.read_excel(DATASET_DIR / "Crowd_Predictor_Dataset_1000_Records.xlsx")

# ==========================
# Encode Columns
# ==========================

categorical_columns = [
    "Destination",
    "Month",
    "Festival_Season",
    "Weekend",
    "School_Holiday",
    "Weather",
    "Crowd_Level"
]

label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ==========================
# Features
# ==========================

X = df[
    [
        "Destination",
        "Month",
        "Festival_Season",
        "Weekend",
        "School_Holiday",
        "Weather"
    ]
]

# ==========================
# Target
# ==========================

y = df["Crowd_Level"]

# ==========================
# Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train
# ==========================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Accuracy
# ==========================

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

# ==========================
# Save
# ==========================

pickle.dump(model, open(MODEL_OUT_DIR / "crowd_model.pkl", "wb"))
pickle.dump(label_encoders, open(MODEL_OUT_DIR / "crowd_label_encoders.pkl", "wb"))

print("Done!")