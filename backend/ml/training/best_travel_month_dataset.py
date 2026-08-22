from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Resolve dataset/model directories relative to this file so the script
# works no matter where it's run from.
DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets"
MODEL_OUT_DIR = Path(__file__).resolve().parent.parent / "models"

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv(DATASET_DIR / "AI_Best_Travel_Month_Dataset_1000.csv")

# ==========================
# Label Encoding
# ==========================
encoders = {}

categorical_columns = [
    "Destination",
    "Budget",
    "Trip_Type",
    "Preferred_Weather",
    "Season",
    "Crowd_Level",
    "Hotel_Price_Level",
    "Best_Month"
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ==========================
# Features and Target
# ==========================
X = df.drop("Best_Month", axis=1)
y = df["Best_Month"]

# ==========================
# Train Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train Random Forest
# ==========================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Prediction
# ==========================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# ==========================
# Save Model
# ==========================
joblib.dump(model, MODEL_OUT_DIR / "best_month_model.pkl")
joblib.dump(encoders, MODEL_OUT_DIR / "label_encoders.pkl")

print("\nModel Saved Successfully!")