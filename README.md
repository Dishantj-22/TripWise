# TripWise

An AI-powered trip planning / expense management prototype: a FastAPI backend
serving trip, expense, and ML-prediction endpoints, plus a small set of
static HTML frontend pages.

## Project structure

```
TripWise/
├── backend/
│   ├── app/                     # FastAPI application code
│   │   ├── main.py              # API routes + model loading
│   │   ├── models.py            # SQLAlchemy ORM models (MySQL)
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── ml/
│   │   ├── datasets/            # Source CSV/XLSX training data
│   │   ├── training/            # Standalone scripts that train each model
│   │   └── models/              # Trained model artifacts (*.pkl) served by the API
│   ├── data/
│   │   └── tripwise.db          # Legacy SQLite file (see note below)
│   └── requirements.txt
└── frontend/
    ├── index.html                # Main app UI
    ├── auth.html                 # Login / signup page
    └── vibranteindex.html        # Alternate/experimental UI variant
```


## Running the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Note: `app/models.py` currently points at a hardcoded local MySQL connection
string (`mysql+pymysql://root:123456789@localhost:3306/tripwise`). Update
this (ideally via an environment variable) before running elsewhere.

`backend/data/tripwise.db` is a SQLite file left over from the archive.
The current code connects to MySQL, not SQLite, so this file is unused by
`app/main.py` as it stands — kept here in case it's needed for reference or
a future SQLite fallback.

## Running the frontend

The HTML files in `frontend/` are plain static pages with no build step —
open them directly in a browser, or serve the folder with any static file
server (the backend's CORS settings already allow `localhost:3000` and
`localhost:5500`).
