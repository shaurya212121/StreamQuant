from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import sys
from pathlib import Path
import xgboost as xgb
import pandas as pd
from celery.result import AsyncResult
import redis 
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import SessionLocal
# Added StockNewsSentiment model import
from db.models import StockPrice, StockNewsSentiment 
# Added run_batch_sentiment import
from backend.worker import celery_app, run_heavy_ml_model, run_batch_sentiment

import os
from dotenv import load_dotenv

# Explicitly load root .env file
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

app = FastAPI(title="StreamQuant API", version="1.0")

# Connect to Redis using environment variable
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL)

# In-memory cache to load model files once into RAM instead of reading from disk on every HTTP request
LOADED_MODELS = {}

def get_loaded_model(ticker: str):
    ticker = ticker.upper()
    if ticker not in LOADED_MODELS:
        model_path = Path(__file__).resolve().parent.parent / "ml_models" / f"{ticker}_xgb_model.json"
        if not model_path.exists():
            return None
        m = xgb.XGBClassifier()
        m.load_model(str(model_path))
        LOADED_MODELS[ticker] = m
    return LOADED_MODELS[ticker]

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

# FLAW #11 FIX: Non-blocking DB lookup for FinBERT Sentiment
@app.get("/api/sentiment/{ticker}")
def get_stock_sentiment(ticker: str, db: Session = Depends(get_db)):
    """
    Fast, non-blocking DB lookup. Returns pre-computed FinBERT sentiment in ~2ms.
    Replaces the previous architecture where ML inference blocked the web request.
    """
    ticker = ticker.upper()
    
    # Query the pre-calculated result written by the background Celery worker
    record = (
        db.query(StockNewsSentiment)
        .filter(StockNewsSentiment.ticker == ticker)
        .order_by(StockNewsSentiment.sentiment_date.desc())
        .first()
    )
    
    if not record:
        # Trigger background calculation if missing, but do not block the thread
        run_batch_sentiment.delay(ticker)
        raise HTTPException(
            status_code=404, 
            detail=f"Sentiment data not yet computed for {ticker}. Background job triggered."
        )
        
    return {
        "ticker": ticker,
        "sentiment_score": record.sentiment_score,
        "positive_prob": record.positive_prob,
        "negative_prob": record.negative_prob,
        "neutral_prob": record.neutral_prob,
        "headline_count": record.headline_count,
        "sentiment_date": str(record.sentiment_date),
        "cached_db_result": True
    }

@app.post("/api/sentiment/{ticker}/refresh")
def refresh_stock_sentiment(ticker: str):
    """Manually push a symbol to the Celery queue for an out-of-cycle ML sentiment update."""
    task = run_batch_sentiment.delay(ticker.upper())
    return {
        "message": f"Background sentiment analysis triggered for {ticker.upper()}.",
        "task_id": task.id
    }

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
    ticker = ticker.upper()
    cache_key = f"prediction:{ticker}"

    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            print(f"[CACHE HIT] Returning cached prediction for {ticker} directly from Redis!")
            return json.loads(cached_data)
    except Exception as e:
        print(f"[CACHE WARNING] Redis lookup failed: {e}")

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
        
    model = get_loaded_model(ticker)
    if not model:
        raise HTTPException(
            status_code=404, 
            detail=f"No trained model found for {ticker}."
        )

    features = pd.DataFrame([{
        "sma_ratio": latest_record.sma_ratio or 0.0,
        "macd": latest_record.macd or 0.0,
        "rsi": latest_record.rsi or 50.0,
        "daily_return": latest_record.daily_return or 0.0,
        "volume_z_score": latest_record.volume_z_score or 0.0,
        "rolling_sharpe": getattr(latest_record, "rolling_sharpe", 0.0) or 0.0,
        "rolling_max_drawdown": getattr(latest_record, "rolling_max_drawdown", 0.0) or 0.0,
        "atr_20": getattr(latest_record, "atr_20", 0.0) or 0.0
    }])

    pred = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[pred] * 100)

    is_anomaly = bool(latest_record.volume_z_score and latest_record.volume_z_score > 3.0)
    
    result = {
        "ticker": ticker,
        "latest_close": latest_record.close,
        "trend_prediction": "BULLISH" if pred == 1 else "BEARISH",
        "confidence_pct": round(confidence, 2),
        "volume_z_score": latest_record.volume_z_score,
        "is_volume_anomaly": is_anomaly,
        "date": str(latest_record.date),
        "cached": False
    }

    try:
        cached_payload = result.copy()
        cached_payload["cached"] = True
        redis_client.setex(cache_key, 300, json.dumps(cached_payload))
        print(f"[CACHE SAVED] Saved {ticker} prediction in Redis for 300 seconds.")
    except Exception as e:
        print(f"[CACHE WARNING] Redis save failed: {e}")

    return result

@app.get("/api/tasks/{task_id}")
def get_task_status(task_id: str):
    """
    Check the live status and result of any background Celery ML task.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
    }
    
    if task_result.status == "SUCCESS":
        response["result"] = task_result.result
    elif task_result.status == "FAILURE":
        response["error"] = str(task_result.result)
        
    return response