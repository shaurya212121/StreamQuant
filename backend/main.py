from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import sys
from pathlib import Path
from backend.worker import run_heavy_ml_model
import xgboost as xgb
import pandas as pd
from celery.result import AsyncResult
from backend.worker import celery_app

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import SessionLocal
from db.models import StockPrice
from backend.worker import run_heavy_ml_model

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

@app.get("/api/stocks/{ticker}/prediction")
def get_stock_prediction(ticker: str, db: Session = Depends(get_db)):
    """
    Synchronous Endpoint: Instantly loads Chaiti's XGBoost model and returns the forecast.
    """
    ticker = ticker.upper()
    model_path = Path(__file__).resolve().parent.parent / "ml_models" / f"{ticker}_xgb_model.json"
    
    if not model_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"No trained model found for {ticker}. Available models: AAPL, AMZN, GOOGL, MSFT, NVDA, TSLA"
        )
    latest_record = (
        db.query(StockPrice)
        .filter(StockPrice.ticker == ticker)
        .order_by(StockPrice.date.desc())
        .first()
    )
    if not latest_record:
        raise HTTPException(
            status_code=404, 
            detail=f"No stock data found in database for {ticker}."
        )
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    features = pd.DataFrame([{
        "sma_ratio": latest_record.sma_ratio,
        "macd": latest_record.macd,
        "rsi": latest_record.rsi,
        "daily_return": latest_record.daily_return
    }])

    pred = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[pred] * 100)

    is_anomaly = bool(latest_record.volume_z_score and latest_record.volume_z_score > 3.0)
    
    # 7. Return the forecast
    return {
        "ticker": ticker,
        "latest_close": latest_record.close,
        "trend_prediction": "BULLISH" if pred == 1 else "BEARISH",
        "confidence_pct": round(confidence, 2),
        "volume_z_score": latest_record.volume_z_score,
        "is_volume_anomaly": is_anomaly,
        "date": str(latest_record.date)
    }

@app.get("/api/tasks/{task_id}")
def get_task_status(task_id: str):
    """
    Check the live status and result of any background Celery ML task.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,  # PENDING, STARTED, SUCCESS, FAILURE
    }
    
    if task_result.status == "SUCCESS":
        response["result"] = task_result.result
    elif task_result.status == "FAILURE":
        response["error"] = str(task_result.result)
        
    return response