from pydantic import BaseModel
from typing import Optional, List
from pydantic import BaseModel


# --- User Validation ---
class UserCreate(BaseModel):
    name: str
    email: str
    password: str

# --- Trip Validation ---
class TripCreate(BaseModel):
    source: str
    destination: str
    days: Optional[int] = None
    budget: float
    travellers: int
    vehicle: str

class TripResponse(TripCreate):
    trip_id: int
    class Config:
        from_attributes = True

# --- Expense Validation ---
class ExpenseCreate(BaseModel):
    trip_id: int
    category: str
    amount: float
    date: str
    note: Optional[str] = None

class ExpenseResponse(ExpenseCreate):
    expense_id: int
    class Config:
        from_attributes = True

# --- Specialized Combined Outputs ---
class DashboardResponse(BaseModel):
  trip_details: TripResponse
  total_budget: float
  total_spent: float
  remaining_budget: float
  fuel_cost: float
  category_distribution: dict
  expenses: List[ExpenseResponse] = []  # <-- ADD THIS LINE

  class Config:
    from_attributes = True

class ReportResponse(BaseModel):
    trip_id: int
    destination: str
    total_spent: float
    budget_status: str  # "Under Budget" or "Over Budget"
    per_person_share: float
    detailed_expenses: List[ExpenseResponse]
    
    # --- Mileage Prediction ---
class MileagePredictionRequest(BaseModel):
    Brand: str
    Model: str
    Vehicle_Year: int
    Vehicle_Type: str
    Fuel_Type: str
    Transmission: str
    Seating_Capacity: int
    Odometer_km: float
    Vehicle_Condition: str
    
class ExpensePredictionRequest(BaseModel):
    source: str
    destination: str
    distance_km: float
    travelers: int
    days: int
    vehicle_type: str
    
class BestMonthPredictionRequest(BaseModel):
    destination: str
    budget: str
    trip_type: str
    preferred_weather: str
    days: int
    season: str
    estimated_cost_inr: float
    crowd_level: str
    hotel_price_level: str
    
class CrowdPredictionRequest(BaseModel):
    Destination: str
    Month: str
    Festival_Season: str
    Weekend: str
    School_Holiday: str
    Weather: str
    
class BudgetBreakdownPredictionRequest(BaseModel):
    Destination: str
    Days: int
    Travellers: int
    Trip_Type: str
    Vehicle_Type: str
    Budget_Level: str
    Month: str