from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import sys
import threading
from pathlib import Path
import xgboost as xgb
import pandas as pd
from celery.result import AsyncResult
import redis 
import json
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Depends, HTTPException, Request
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

# --- DISTRIBUTED RATE LIMITER (Redis Sliding Window) ---
RATE_LIMIT_MAX_REQUESTS = 30  # Max requests allowed
RATE_LIMIT_WINDOW_SECONDS = 60  # Time window (1 minute)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Only rate-limit API endpoints, skip docs and health checks
    if request.url.path.startswith("/api/"):
        client_ip = request.client.host if request.client else "unknown"
        rate_key = f"rate_limit:{client_ip}"

        try:
            # Atomic increment in Redis
            requests_count = redis_client.incr(rate_key)
            if requests_count == 1:
                # Set 60-second expiration on the first request
                redis_client.expire(rate_key, RATE_LIMIT_WINDOW_SECONDS)

            if requests_count > RATE_LIMIT_MAX_REQUESTS:
                ttl = redis_client.ttl(rate_key)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Too Many Requests",
                        "message": f"Rate limit exceeded (30 req/min). Try again in {ttl} seconds."
                    }
                )
        except Exception as e:
            # Fail-open: If Redis is down, don't block legitimate user traffic
            pass

    response = await call_next(request)
    return response
# Connect to Redis using environment variable
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL)

# In-memory cache to load model files once into RAM instead of reading from disk on every HTTP request
LOADED_MODELS = {}
_model_cache_lock = threading.Lock()

def get_loaded_model(ticker: str):
    ticker = ticker.upper()
    if ticker in LOADED_MODELS:
        return LOADED_MODELS[ticker]
    with _model_cache_lock:
        if ticker not in LOADED_MODELS:
            model_path = Path(__file__).resolve().parent.parent / "ml_models" / f"{ticker}_xgb_model.json"
            if not model_path.exists():
                return None
            m = xgb.XGBClassifier()
            m.load_model(str(model_path))
            LOADED_MODELS[ticker] = m
    return LOADED_MODELS.get(ticker)

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
        .order_by(StockNewsSentiment.date.desc())
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

# =====================================================================
# UNIFIED INTELLIGENCE GATEWAY (Aggregated AI + Quant + Sentiment)
# =====================================================================
@app.get("/api/stocks/{ticker}/intelligence")
def get_unified_stock_intelligence(ticker: str, db: Session = Depends(get_db)):
    """
    Unified Intelligence Gateway:
    Consolidates XGBoost trend prediction, SHAP feature drivers, Volume Z-Score 
    anomaly alerts, and FinBERT news sentiment into a single Redis-cached response.
    """
    ticker = ticker.upper()
    cache_key = f"intelligence:{ticker}"

    # 1. Check Upstash Redis Cache First (< 10ms response)
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            print(f"[CACHE HIT] Returning unified intelligence for {ticker} from Redis!")
            return json.loads(cached_data)
    except Exception as e:
        print(f"[CACHE WARNING] Redis lookup failed: {e}")

    # 2. Query Latest Price & Technical Features from PostgreSQL
    latest_price = (
        db.query(StockPrice)
        .filter(StockPrice.ticker == ticker)
        .order_by(StockPrice.date.desc())
        .first()
    )
    if not latest_price:
        raise HTTPException(
            status_code=404,
            detail=f"No price records found for ticker '{ticker}' in database."
        )

    # 3. Load Trained XGBoost Model
    model = get_loaded_model(ticker)
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"No trained XGBoost model found for '{ticker}'. Available: AAPL, AMZN, GOOGL, MSFT, NVDA, TSLA."
        )

    # 4. Construct Feature Vector (All 8 Features)
    feature_cols = [
        "sma_ratio", "macd", "rsi", "daily_return",
        "volume_z_score", "rolling_sharpe", "rolling_max_drawdown", "atr_20"
    ]
    features_df = pd.DataFrame([{
        "sma_ratio": latest_price.sma_ratio or 0.0,
        "macd": latest_price.macd or 0.0,
        "rsi": latest_price.rsi or 50.0,
        "daily_return": latest_price.daily_return or 0.0,
        "volume_z_score": latest_price.volume_z_score or 0.0,
        "rolling_sharpe": getattr(latest_price, "rolling_sharpe", 0.0) or 0.0,
        "rolling_max_drawdown": getattr(latest_price, "rolling_max_drawdown", 0.0) or 0.0,
        "atr_20": getattr(latest_price, "atr_20", 0.0) or 0.0
    }])

    # 5. Execute XGBoost Inference
    pred = int(model.predict(features_df)[0])
    probabilities = model.predict_proba(features_df)[0]
    confidence_pct = round(float(probabilities[pred] * 100), 2)
    is_bullish = (pred == 1)

    # 6. Compute SHAP Top-3 Feature Drivers (Explainability)
    top_drivers = []
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(features_df)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = shap_vals[0] if shap_vals.ndim > 1 else shap_vals
        
        feature_vals = features_df.iloc[0]
        contributions = []
        for name, val, s_val in zip(feature_cols, feature_vals, shap_vals):
            contributions.append({
                "feature": name,
                "current_value": round(float(val), 4),
                "shap_impact": round(float(s_val), 4)
            })
        contributions.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
        top_drivers = contributions[:3]
    except Exception as e:
        print(f"[SHAP NOTICE] Fallback used: {e}")
        # Graceful fallback: return top features by raw input values if SHAP is absent
        top_drivers = [
            {"feature": "rsi", "current_value": round(float(latest_price.rsi or 50), 2), "shap_impact": 0.0},
            {"feature": "sma_ratio", "current_value": round(float(latest_price.sma_ratio or 0), 4), "shap_impact": 0.0},
            {"feature": "volume_z_score", "current_value": round(float(latest_price.volume_z_score or 0), 2), "shap_impact": 0.0}
        ]

    # 7. Evaluate Statistical Volume Anomaly (|Z| > 3.0)
    vol_z = latest_price.volume_z_score or 0.0
    is_anomaly = bool(vol_z > 3.0)
    anomaly_alert = (
        f"WHALE TRADE DETECTED: Volume is +{vol_z:.2f}σ above 20-day mean" 
        if is_anomaly else "Normal institutional liquidity"
    )

    # 8. Query Pre-Computed FinBERT News Sentiment
    sentiment_record = (
        db.query(StockNewsSentiment)
        .filter(StockNewsSentiment.ticker == ticker)
        .order_by(StockNewsSentiment.date.desc())
        .first()
    )
    if sentiment_record:
        sentiment_data = {
            "sentiment_score": round(float(sentiment_record.sentiment_score), 4),
            "positive_prob": round(float(sentiment_record.positive_prob), 4),
            "negative_prob": round(float(sentiment_record.negative_prob), 4),
            "neutral_prob": round(float(sentiment_record.neutral_prob), 4),
            "headline_count": sentiment_record.headline_count,
            "status": "AVAILABLE"
        }
    else:
        # Trigger background Celery calculation if missing
        run_batch_sentiment.delay(ticker)
        sentiment_data = {
            "sentiment_score": 0.0,
            "positive_prob": 0.33,
            "negative_prob": 0.33,
            "neutral_prob": 0.34,
            "headline_count": 0,
            "status": "PROCESSING_IN_BACKGROUND"
        }

    # 9. Compute Unified Confluence Signal (Combining ML + NLP + Volume)
    news_score = sentiment_data["sentiment_score"]
    if is_bullish and news_score > 0.15 and not is_anomaly:
        signal = "STRONG BUY"
        summary = "Technical trend is Bullish with supportive news sentiment and stable liquidity."
    elif is_bullish and news_score < -0.15:
        signal = "CAUTIOUS BUY (DIVERGENCE)"
        summary = "Model predicts upward trend, but news sentiment is Negative. Watch for headline risk."
    elif not is_bullish and news_score < -0.15:
        signal = "STRONG SELL"
        summary = "Technical trend is Bearish aligned with negative financial news sentiment."
    elif not is_bullish and news_score > 0.15:
        signal = "CAUTIOUS SELL (DIVERGENCE)"
        summary = "Technical trend is Bearish despite positive news sentiment."
    elif is_anomaly:
        signal = "HIGH VOLATILITY ALERT"
        summary = f"Volume spike detected ({vol_z:.2f}σ). Institutional block trades in progress."
    else:
        signal = "NEUTRAL / HOLD"
        summary = "Mixed or balanced indicators across technical and sentiment layers."

    # 10. Construct Final Unified Payload
    response_payload = {
        "ticker": ticker,
        "as_of_date": str(latest_price.date),
        "latest_close": round(float(latest_price.close), 2),
        "confluence_decision": {
            "signal": signal,
            "summary": summary
        },
        "ml_trend_layer": {
            "prediction": "BULLISH" if is_bullish else "BEARISH",
            "confidence_pct": confidence_pct,
            "top_drivers": top_drivers
        },
        "statistical_anomaly_layer": {
            "volume_z_score": round(float(vol_z), 2),
            "is_anomaly": is_anomaly,
            "alert": anomaly_alert
        },
        "nlp_sentiment_layer": sentiment_data,
        "cached": False
    }

    # 11. Save in Redis with 5-minute (300s) TTL
    try:
        redis_payload = response_payload.copy()
        redis_payload["cached"] = True
        redis_client.setex(cache_key, 300, json.dumps(redis_payload))
        print(f"[CACHE SAVED] Cached unified intelligence for {ticker} (TTL: 300s).")
    except Exception as e:
        print(f"[CACHE WARNING] Failed to cache in Redis: {e}")

    return response_payload
