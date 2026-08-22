from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

# MySQL Connection config
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:123456789@localhost:3306/tripwise"

engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id=Column(Integer, primary_key=True, index=True)
    name=Column(String(100), nullable=False)
    email=Column(String(150), unique=True, nullable=False)
    password=Column(String(200), nullable=False)
    
class Trip(Base):
    __tablename__ = "trips"
    
    trip_id=Column(Integer, primary_key=True, index=True)
    source=Column(String(100), nullable=False)
    destination=Column(String(100), nullable=False)
    days=Column(Integer, nullable=True)
    budget=Column(Float, nullable=False)
    travellers=Column(Integer, nullable=False)
    vehicle=Column(String(50), nullable=False)
    created_at=Column(DateTime, server_default=func.now())
    
class Expense(Base):
    __tablename__="expenses"
    
    expense_id=Column(Integer, primary_key=True, index=True, autoincrement=True)
    trip_id=Column(Integer, ForeignKey("trips.trip_id"), nullable=False)
    category=Column(String(50), nullable=False)
    amount=Column(Float, nullable=False)
    date=Column(String(20), nullable=False)
    note=Column(String(100), nullable=True)