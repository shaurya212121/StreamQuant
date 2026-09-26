from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import sys
from pathlib import Path
from backend.worker import run_heavy_ml_model

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import SessionLocal
from db.models import StockPrice

app = FastAPI(title="StreamQuant API", version="1.0")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def health_check():
    return {"status": "Online", "message": "StreamQuant API is ready to serve data!"}

@app.get("/api/stocks/{ticker}")
def get_stock_data(ticker: str, limit: int = 30, db: Session = Depends(get_db)):
        records = (
        db.query(StockPrice)
        .filter(StockPrice.ticker == ticker.upper())
        .order_by(StockPrice.date.desc())
        .limit(limit)
        .all()
    )

        if not records:
            raise HTTPException(
            status_code=404, 
        detail=f"No data found in the database for ticker {ticker}."
            )

        return {"ticker": ticker.upper(), "data": records}

@app.post("/api/ml-task/{ticker}")
def start_ml_task(ticker: str):
    """
    This pushes the heavy ML task to Redis and immediately replies to the user.
    """
    # .delay() is the magic Celery word that means "Send this to Redis!"
    task = run_heavy_ml_model.delay(ticker.upper())
    
    return {
        "message": f"ML Task for {ticker.upper()} has been sent to the background!",
        "task_id": task.id
    }