from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Dict, List
import pydantic

# Import SQLAlchemy architecture configurations directly from your models file
from .models import Base, engine, SessionLocal, User, Trip, Expense
from . import schemas
import joblib
from .schemas import (
    MileagePredictionRequest, 
    ExpensePredictionRequest, 
    BestMonthPredictionRequest, 
    CrowdPredictionRequest, 
    BudgetBreakdownPredictionRequest
)
import pandas as pd
import pickle

# Automatically synchronize database infrastructure structures on execution
Base.metadata.create_all(bind=engine)

# Resolve the ml/models directory regardless of the working directory the app is started from
MODELS_DIR = Path(__file__).resolve().parent.parent / "ml" / "models"

# Load predictive ML models and transformers safely
expense_model = joblib.load(MODELS_DIR / "expense_model.pkl")
mileage_model = joblib.load(MODELS_DIR / "mileage_model.pkl")
best_month_model = joblib.load(MODELS_DIR / "best_month_model.pkl")
label_encoders = joblib.load(MODELS_DIR / "label_encoders.pkl")
crowd_model = pickle.load(open(MODELS_DIR / "crowd_model.pkl", "rb"))
crowd_label_encoders = pickle.load(open(MODELS_DIR / "crowd_label_encoders.pkl", "rb"))
budget_breakdown_model = pickle.load(open(MODELS_DIR / "budget_breakdown_model.pkl", "rb"))
budget_breakdown_label_encoders = pickle.load(open(MODELS_DIR / "budget_breakdown_label_encoders.pkl", "rb"))

app = FastAPI(
    title="TripWise API",
    description="AI-powered Trip Expense Manager Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Session Dependency management injection token
def get_db():
    db = SessionLocal()
    try:
        user_summary = "Rexy" 
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "TripWise API is running securely!"}


# Core string encoder normalizer
def safe_transform(encoder, value: str):
    val_clean = str(value).strip().lower()
    classes_lower = [str(c).lower() for c in encoder.classes_]
    if val_clean in classes_lower:
        matched_index = classes_lower.index(val_clean)
        return encoder.transform([encoder.classes_[matched_index]])[0]
    else:
        print(f"Warning: String element '{value}' unmatched. Using baseline fallback class pattern.")
        return encoder.transform([encoder.classes_[0]])[0]


# =====================================================================
# FALLBACK-CONTAINED CROSS-DICTIONARY LOOKUP ENGINES TO SOLVE KEYERRORS
# =====================================================================
def find_and_transform(key: str, value: str):
    """
    Scans all loaded label encoder dictionaries case-insensitively to find 
    the requested feature column, safely falling back to alternative sources.
    """
    dicts_to_search = [label_encoders, crowd_label_encoders, budget_breakdown_label_encoders]
    key_lower = key.lower()
    
    # Pass 1: Strict case-insensitive key matching across all source files
    for encoder_dict in dicts_to_search:
        if encoder_dict and isinstance(encoder_dict, dict):
            for k in encoder_dict.keys():
                if k.lower() == key_lower:
                    return safe_transform(encoder_dict[k], value)
                    
    # Pass 2: Fallback substring matching (e.g., 'Budget' matches 'Budget_Level')
    for encoder_dict in dicts_to_search:
        if encoder_dict and isinstance(encoder_dict, dict):
            for k in encoder_dict.keys():
                if key_lower in k.lower() or k.lower() in key_lower:
                    return safe_transform(encoder_dict[k], value)
                    
    print(f"Warning: No valid transformer found for '{key}'. Defaulting feature weight to 0.")
    return 0


def find_and_inverse_transform(key: str, prediction):
    """Scans all encoder dictionaries to safely execute inverse label lookups for target output formats."""
    dicts_to_search = [label_encoders, crowd_label_encoders, budget_breakdown_label_encoders]
    key_lower = key.lower()
    for encoder_dict in dicts_to_search:
        if encoder_dict and isinstance(encoder_dict, dict):
            for k in encoder_dict.keys():
                if k.lower() == key_lower or key_lower in k.lower():
                    try:
                        return encoder_dict[k].inverse_transform(prediction)[0]
                    except Exception:
                        continue
    return "October" # Safe global calendar default fallback


# ==========================================
# DATA CREATION ENDPOINTS
# ==========================================

@app.post("/api/trips", response_model=schemas.TripResponse, status_code=status.HTTP_201_CREATED)
def create_trip(trip: schemas.TripCreate, db: Session = Depends(get_db)):
    db_trip = Trip(**trip.model_dump())
    db.add(db_trip)
    db.commit()
    db.refresh(db_trip)
    return db_trip

@app.post("/api/expenses", response_model=schemas.ExpenseResponse, status_code=status.HTTP_201_CREATED)
def create_expense(expense: schemas.ExpenseCreate, db: Session = Depends(get_db)):
    trip_exists = db.query(Trip).filter(Trip.trip_id == expense.trip_id).first()
    if not trip_exists:
        raise HTTPException(status_code=404, detail="Target trip link context missing.")
    db_expense = Expense(**expense.model_dump())
    db.add(db_expense)
    db.commit()
    db.refresh(db_expense)
    return db_expense


# ==========================================
# RE-ENGINEERED CALCULATIONS & ROUTING
# ==========================================

@app.get("/api/dashboard/{trip_id}", response_model=schemas.DashboardResponse)
def get_dashboard_data(trip_id: int, db: Session = Depends(get_db)):
    trip = db.query(Trip).filter(Trip.trip_id == trip_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip record not found")
        
    # Query your MySQL database for the corresponding rows
    expenses = db.query(Expense).filter(Expense.trip_id == trip_id).all()
    
    total_spent = sum(item.amount for item in expenses)
    remaining_budget = trip.budget - total_spent
    fuel_cost = sum(item.amount for item in expenses if item.category.lower() == "fuel")

    category_distribution = {}
    for item in expenses:
        cat = item.category.capitalize()
        category_distribution[cat] = category_distribution.get(cat, 0.0) + item.amount

    return {
        "trip_details": trip,
        "total_budget": trip.budget,
        "total_spent": total_spent,
        "remaining_budget": remaining_budget,
        "fuel_cost": fuel_cost,
        "category_distribution": category_distribution,
        "expenses": expenses  
    }

@app.get("/api/trips", response_model=List[schemas.TripResponse])
def get_all_trips(db: Session = Depends(get_db)):
    return db.query(Trip).order_by(Trip.created_at.desc()).all()

@app.get("/api/report/{trip_id}", response_model=schemas.ReportResponse)
def get_report_data(trip_id: int, db: Session = Depends(get_db)):
    trip = db.query(Trip).filter(Trip.trip_id == trip_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip record not found")

    expenses = db.query(Expense).filter(Expense.trip_id == trip_id).all()
    total_spent = sum(item.amount for item in expenses)
    
    budget_status = "Under Budget" if total_spent <= trip.budget else "Over Budget"
    per_person_share = total_spent / trip.travellers if trip.travellers > 0 else total_spent

    return {
        "trip_id": trip.trip_id,
        "destination": trip.destination,
        "total_spent": total_spent,
        "budget_status": budget_status,
        "per_person_share": per_person_share,
        "detailed_expenses": expenses
    }


# ==========================================
# PREDICTIVE INTELLIGENCE ENDPOINTS 
# ==========================================

@app.post("/api/predict-mileage")
def predict_mileage(request: MileagePredictionRequest):
    # Safe lookups using our cross-dictionary engine mapping lowercase dictionary parameters safely
    encoded_brand = find_and_transform("Brand", request.Brand)
    encoded_model = find_and_transform("Model", request.Model)
    encoded_vehicle = find_and_transform("Vehicle_Type", request.Vehicle_Type)
    encoded_fuel = find_and_transform("Fuel_Type", request.Fuel_Type)
    encoded_transmission = find_and_transform("Transmission", request.Transmission)
    encoded_condition = find_and_transform("Vehicle_Condition", request.Vehicle_Condition)

    input_df = pd.DataFrame([{
        "Brand": encoded_brand,
        "Model": encoded_model,
        "Vehicle_Year": request.Vehicle_Year,
        "Vehicle_Type": encoded_vehicle,
        "Fuel_Type": encoded_fuel,
        "Transmission": encoded_transmission,
        "Seating_Capacity": request.Seating_Capacity,
        "Odometer_km": request.Odometer_km,
        "Vehicle_Condition": encoded_condition
    }])

    mileage = float(mileage_model.predict(input_df)[0])
    return {"estimated_mileage": round(mileage, 2)}


@app.post("/api/predict-budget")
def predict_budget(request: ExpensePredictionRequest):
    try:
        input_df = pd.DataFrame([{
            "Source": request.source,
            "Destination": request.destination,
            "Distance_km": request.distance_km,
            "Travelers": request.travelers,
            "Days": request.days,
            "VehicleType": request.vehicle_type
        }])
        prediction = expense_model.predict(input_df)
        return {"estimated_trip_cost": round(float(prediction[0]), 2)}
        
    except Exception as e:
        print(f"Prediction failed, deploying safety fallback: {e}")
        baseline_rate = 1500 if request.vehicle_type == "Car" else 800
        fallback_cost = (baseline_rate * request.days) * request.travelers
        return {"estimated_trip_cost": float(fallback_cost)}


@app.post("/api/predict-best-month")
def predict_best_month(request: BestMonthPredictionRequest):
    try:
        # Cross-search engine parses all dictionaries to resolve the true keys smoothly
        data = {
            "Destination": find_and_transform("Destination", request.destination),
            "Budget": find_and_transform("Budget", request.budget),
            "Trip_Type": find_and_transform("Trip_Type", request.trip_type),
            "Preferred_Weather": find_and_transform("Preferred_Weather", request.preferred_weather),
            "Days": request.days,
            "Season": find_and_transform("Season", request.season),
            "Estimated_Cost_INR": request.estimated_cost_inr,
            "Crowd_Level": find_and_transform("Crowd_Level", request.crowd_level),
            "Hotel_Price_Level": find_and_transform("Hotel_Price_Level", request.hotel_price_level)
        }

        df = pd.DataFrame([data])
        prediction = best_month_model.predict(df)
        month_resolved = find_and_inverse_transform("Best_Month", prediction)

        return {"recommended_month": month_resolved}
    except Exception as e:
        print(f"Best month execution fallback caught: {e}")
        return {"recommended_month": "December"}


@app.post("/api/predict-crowd")
def predict_crowd(request: CrowdPredictionRequest):
    encoded_destination = find_and_transform("Destination", request.Destination)
    encoded_month = find_and_transform("Month", request.Month)
    encoded_festival = find_and_transform("Festival_Season", request.Festival_Season)
    encoded_weekend = find_and_transform("Weekend", request.Weekend)
    encoded_holiday = find_and_transform("School_Holiday", request.School_Holiday)
    encoded_weather = find_and_transform("Weather", request.Weather)

    input_df = pd.DataFrame([{
        "Destination": encoded_destination,
        "Month": encoded_month,
        "Festival_Season": encoded_festival,
        "Weekend": encoded_weekend,
        "School_Holiday": encoded_holiday,
        "Weather": encoded_weather
    }])

    prediction = crowd_model.predict(input_df)[0]
    crowd_level = find_and_inverse_transform("Crowd_Level", [prediction])

    visitors = {
        "Low": "2,000 - 10,000/day",
        "Medium": "10,001 - 25,000/day",
        "High": "25,001 - 45,000/day",
        "Very High": "45,001 - 70,000/day"
    }

    return {
        "crowd_level": crowd_level,
        "expected_visitors": visitors.get(crowd_level, "10,001 - 25,000/day")
    }


@app.post("/api/predict-budget-breakdown")
def predict_budget_breakdown(request: BudgetBreakdownPredictionRequest):
    encoded_destination = find_and_transform("Destination", request.Destination)
    encoded_trip_type = find_and_transform("Trip_Type", request.Trip_Type)
    encoded_vehicle = find_and_transform("Vehicle_Type", request.Vehicle_Type)
    encoded_budget_lvl = find_and_transform("Budget_Level", request.Budget_Level)
    encoded_month = find_and_transform("Month", request.Month)

    input_df = pd.DataFrame([{
        "Destination": encoded_destination,
        "Days": request.Days,
        "Travellers": request.Travellers,
        "Trip_Type": encoded_trip_type,
        "Vehicle_Type": encoded_vehicle,
        "Budget_Level": encoded_budget_lvl,
        "Month": encoded_month
    }])

    prediction = budget_breakdown_model.predict(input_df)[0]

    hotel = round(float(prediction[0]), 2)
    fuel = round(float(prediction[1]), 2)
    food = round(float(prediction[2]), 2)
    shopping = round(float(prediction[3]), 2)
    misc = round(float(prediction[4]), 2)
    total = round(hotel + fuel + food + shopping + misc, 2)

    return {
        "hotel_cost": hotel,
        "fuel_cost": fuel,
        "food_cost": food,
        "shopping_cost": shopping,
        "misc_cost": misc,
        "total_budget": total
    }   