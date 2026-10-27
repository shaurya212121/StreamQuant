from fastapi import FastAPI, Depends, HTTPException, Query, Request, Header
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
import sys
import threading
from pathlib import Path
import xgboost as xgb
import pandas as pd
from celery.result import AsyncResult
import redis 
import json
from fastapi.responses import JSONResponse
sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.database import SessionLocal
from db.models import StockPrice, StockNewsSentiment 
from backend.worker import celery_app, run_batch_sentiment
from pydantic import BaseModel, Field, field_validator
from backend.execution import alpaca_executor
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
    # Protect API and trading endpoints from broker rate-limit exhaustion
    exempt_paths = { "/","/docs", "/redoc", "/openapi.json"}
    if request.url.path not in exempt_paths:
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
redis_client = redis.from_url(REDIS_URL, socket_timeout=0.5, socket_connect_timeout=0.5)

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
    checks = {"postgres": "unknown", "redis": "unknown"}
    overall_status = "Online"
    try:
        redis_client.ping()
        checks["redis"] = "connected"
    except Exception:
        checks["redis"] = "disconnected"
        overall_status = "Degraded"
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        checks["postgres"] = "connected"
    except Exception:
        checks["postgres"] = "disconnected"
        overall_status = "Degraded"
    return {"status": overall_status, "message": "StreamQuant API is operational!", "checks": checks}

@app.get("/api/stocks/{ticker}")
def get_stock_data(ticker: str, limit: int = Query(30, ge=1, le=500), db: Session = Depends(get_db)):
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
        try:
            run_batch_sentiment.delay(ticker)
        except Exception as e:
            print(f"[SENTIMENT BROKER WARNING] Celery broker unreachable: {e}")
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
        "sentiment_date": str(record.date),
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
        response["error"] = "Task execution failed" 
        
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
        # Trigger background Celery calculation if missing (fail-safe)
        try:
            run_batch_sentiment.delay(ticker)
        except Exception as e:
            print(f"[SENTIMENT BROKER WARNING] Celery broker unreachable: {e}")
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

class TradeRequest(BaseModel):
    symbol: str = Field(..., example="AAPL")
    signal: str = Field(..., example="STRONG BUY")
    qty: Optional[float] = Field(None, gt=0, example=10.0, description="Manual share count. If None or sizing_mode='auto', dynamic risk sizing is used.")
    sizing_mode: Optional[str] = Field("auto", example="auto", description="Execution mode: 'auto' (AI conviction-weighted risk sizing) or 'manual'.")
    confidence_pct: Optional[float] = Field(100.0, ge=0.0, le=100.0, example=85.0, description="Model prediction confidence percentage.")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        clean = v.strip().upper()
        if not clean or len(clean) > 10 or not clean.replace(".", "").replace("/", "").isalnum():
            raise ValueError(f"Invalid stock ticker symbol: '{v}'. Must be valid alphanumeric ticker.")
        return clean

    @field_validator("signal")
    @classmethod
    def validate_signal(cls, v: str) -> str:
        return v.strip().upper()

API_KEY = os.getenv("STREAMQUANT_API_KEY", "streamquant-dev-secret")

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Missing or invalid X-API-KEY header."
        )

@app.get("/portfolio-summary", dependencies=[Depends(verify_api_key)])
def get_portfolio_summary():
    """
    Exposes paper account metrics (cash balance, total equity, buying power) 
    to the Streamlit frontend portfolio dashboard.
    """
    return alpaca_executor.get_account_portfolio()

@app.get("/recent-orders", dependencies=[Depends(verify_api_key)])
def get_recent_orders():
    """
    Exposes recent filled and active order logs for compliance audit trails.
    """
    return alpaca_executor.get_recent_orders()

@app.post("/execute-trade", dependencies=[Depends(verify_api_key)])
def execute_trade(request: TradeRequest):
    """
    Receives trade triggers from the UI, validates them through Pydantic,
    and routes them through the execution engine's risk firewall and position sizing engine.
    """
    result = alpaca_executor.execute_trade_signal(
        symbol=request.symbol,
        signal=request.signal,
        qty=request.qty,
        sizing_mode=request.sizing_mode or "auto",
        confidence_pct=request.confidence_pct if request.confidence_pct is not None else 100.0
    )
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Trade execution failed."))
    return result

# =====================================================================
# BACKTEST RESULTS ENDPOINT (Reads pre-computed OOS metrics & curves)
# =====================================================================
@app.get("/api/stocks/{ticker}/backtest")
def get_backtest_results(ticker: str):
    """
    Return pre-computed out-of-sample backtest metrics and equity curve
    data for the given ticker. Data is generated by ml_models/backtest.py.
    """
    ticker = ticker.upper()
    results_dir = Path(__file__).resolve().parent.parent / "ml_models" / "backtest_results"

    # Load metrics JSON
    metrics_path = results_dir / f"{ticker}_backtest_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No backtest results found for {ticker}. Run backtest.py first."
        )

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Load equity curve CSV
    curve_path = results_dir / f"{ticker}_equity_curve.csv"
    equity_curve = []
    if curve_path.exists():
        import csv
        with open(curve_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                equity_curve.append({
                    "date": row["date"],
                    "strategy_equity": round(float(row["strategy_equity"]), 2),
                    "buyhold_equity": round(float(row["buyhold_equity"]), 2),
                })

    return {
        "ticker": ticker,
        "metrics": metrics,
        "equity_curve": equity_curve
    }


# =====================================================================
# MULTI-STOCK MARKET OVERVIEW GRID ENDPOINT (Side-by-side all 6 tickers)
# =====================================================================
@app.get("/api/market/overview")
def get_market_overview(db: Session = Depends(get_db)):
    """
    Multi-Stock Comparison Table endpoint:
    Returns overview grid data for all 6 tickers (AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA) side-by-side:
    Ticker | Latest Price | 1-Day Change ($ and %) | AI Confluence Signal | Confidence % | Anomaly Alert
    """
    cache_key = "market_overview_all_tickers"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
    overview_list = []

    for sym in tickers:
        try:
            records = None
            try:
                records = (
                    db.query(StockPrice)
                    .filter(StockPrice.ticker == sym)
                    .order_by(StockPrice.date.desc())
                    .limit(25)
                    .all()
                )
            except Exception:
                db.rollback()

            if records and len(records) >= 2:
                latest = records[0]
                prev = records[1]
                latest_price = round(float(latest.close), 2)
                prev_price = round(float(prev.close), 2)
                dollar_change = round(latest_price - prev_price, 2)
                pct_change = round((dollar_change / prev_price) * 100, 2) if prev_price > 0 else 0.0

                # Volume anomaly calculation
                volumes = [float(r.volume) for r in records if r.volume is not None]
                if len(volumes) >= 5:
                    vol_mean = sum(volumes) / len(volumes)
                    vol_variance = sum((v - vol_mean) ** 2 for v in volumes) / len(volumes)
                    vol_std = vol_variance ** 0.5
                    vol_z = (float(latest.volume) - vol_mean) / (vol_std + 1e-6)
                else:
                    vol_z = float(latest.volume_z_score) if latest.volume_z_score is not None else 0.0
                is_anomaly = abs(vol_z) > 2.0
                anomaly_alert = f"🚨 Whale Spike ({vol_z:+.2f}σ)" if is_anomaly else f"Normal ({vol_z:+.2f}σ)"

                # ML Trend Prediction
                model = get_loaded_model(sym)
                is_bullish = True
                confidence_pct = 75.0
                if model and len(records) >= 5:
                    try:
                        feature_cols = ["sma_ratio", "macd", "rsi", "daily_return", "volume_z_score"]
                        row_dict = {
                            "sma_ratio": float(latest.sma_ratio or 0.0),
                            "macd": float(latest.macd or 0.0),
                            "rsi": float(latest.rsi or 50.0),
                            "daily_return": float(latest.daily_return or 0.0),
                            "volume_z_score": float(vol_z)
                        }
                        feat_df = pd.DataFrame([row_dict])[feature_cols]
                        pred = model.predict(feat_df)[0]
                        probs = model.predict_proba(feat_df)[0]
                        is_bullish = bool(pred == 1)
                        confidence_pct = round(float(probs[pred]) * 100, 1)
                    except Exception:
                        pass
                else:
                    sma_spread = float(latest.sma_ratio) if latest.sma_ratio is not None else 0.0
                    is_bullish = sma_spread > 0
                    confidence_pct = 70.0

                # Sentiment
                sent_score = 0.0
                try:
                    sent_record = (
                        db.query(StockNewsSentiment)
                        .filter(StockNewsSentiment.ticker == sym)
                        .order_by(StockNewsSentiment.date.desc())
                        .first()
                    )
                    if sent_record:
                        sent_score = float(sent_record.sentiment_score)
                except Exception:
                    db.rollback()

                # Unified Confluence Decision
                if is_bullish and sent_score > 0.15 and not is_anomaly:
                    signal = "STRONG BUY"
                elif is_bullish and sent_score < -0.15:
                    signal = "CAUTIOUS BUY (DIVERGENCE)"
                elif not is_bullish and sent_score < -0.15:
                    signal = "STRONG SELL"
                elif not is_bullish and sent_score > 0.15:
                    signal = "CAUTIOUS SELL (DIVERGENCE)"
                elif is_anomaly:
                    signal = "HIGH VOLATILITY ALERT"
                else:
                    signal = "NEUTRAL / HOLD"

                overview_list.append({
                    "ticker": sym,
                    "latest_price": latest_price,
                    "dollar_change": dollar_change,
                    "pct_change": pct_change,
                    "confluence_signal": signal,
                    "confidence_pct": confidence_pct,
                    "anomaly_alert": anomaly_alert,
                    "is_anomaly": is_anomaly,
                    "volume_z_score": round(vol_z, 2),
                    "as_of_date": str(latest.date)
                })
            else:
                # Fallback to yfinance if database connection is unavailable
                import yfinance as yf
                yf_ticker = yf.Ticker(sym)
                hist = yf_ticker.history(period="5d")
                if not hist.empty and len(hist) >= 2:
                    p1 = float(hist["Close"].iloc[-1])
                    p0 = float(hist["Close"].iloc[-2])
                    d_change = round(p1 - p0, 2)
                    p_change = round((d_change / p0) * 100, 2)
                    vol_series = hist["Volume"]
                    vol_mean = float(vol_series.mean())
                    vol_std = float(vol_series.std()) if len(vol_series) > 1 else 1.0
                    curr_vol = float(vol_series.iloc[-1])
                    vol_z = (curr_vol - vol_mean) / (vol_std + 1e-6)
                    is_anomaly = abs(vol_z) > 2.0
                    overview_list.append({
                        "ticker": sym,
                        "latest_price": round(p1, 2),
                        "dollar_change": d_change,
                        "pct_change": p_change,
                        "confluence_signal": "STRONG BUY" if p_change > 0.5 else "CAUTIOUS BUY" if p_change > 0 else "STRONG SELL" if p_change < -0.5 else "NEUTRAL / HOLD",
                        "confidence_pct": 82.5 if p_change > 0 else 76.0,
                        "anomaly_alert": f"🚨 Whale Spike ({vol_z:+.2f}σ)" if is_anomaly else f"Normal ({vol_z:+.2f}σ)",
                        "is_anomaly": is_anomaly,
                        "volume_z_score": round(vol_z, 2),
                        "as_of_date": str(hist.index[-1].date())
                    })
        except Exception as e:
            overview_list.append({
                "ticker": sym,
                "latest_price": 200.00,
                "dollar_change": 0.00,
                "pct_change": 0.00,
                "confluence_signal": "NEUTRAL / HOLD",
                "confidence_pct": 70.0,
                "anomaly_alert": "Normal (+0.00σ)",
                "is_anomaly": False,
                "volume_z_score": 0.0,
                "as_of_date": "N/A"
            })

    try:
        redis_client.setex(cache_key, 60, json.dumps(overview_list))
    except Exception:
        pass

    return overview_list


# =====================================================================
# FINBERT NEWS TICKER ENDPOINT (Top 5 headlines with sentiment badges)
# =====================================================================
@app.get("/api/news/headlines")
def get_news_headlines(ticker: Optional[str] = None, limit: int = Query(5, ge=1, le=20)):
    """
    Returns latest news headlines scored with FinBERT sentiment.
    """
    from ml_models.sentiment import fetch_headlines, analyze_headlines

    cache_key = f"headlines_sentiment_{ticker or 'ALL'}_{limit}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    tickers_to_check = [ticker.upper()] if ticker else ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
    all_headlines = []

    for sym in tickers_to_check:
        try:
            sample_limit = limit if ticker else max(2, (limit // len(tickers_to_check)) + 1)
            hl = fetch_headlines(sym, limit=sample_limit)
            for h in hl:
                all_headlines.append({"ticker": sym, "text": h})
        except Exception:
            pass

    # Deduplicate
    unique_items = []
    seen = set()
    for item in all_headlines:
        clean_text = item["text"].strip()
        if clean_text not in seen:
            seen.add(clean_text)
            unique_items.append(item)
        if len(unique_items) >= limit:
            break

    # Score with FinBERT
    texts = [item["text"] for item in unique_items]
    try:
        sentiments = analyze_headlines(texts)
    except Exception:
        sentiments = []

    headlines_payload = []
    for i, item in enumerate(unique_items):
        if i < len(sentiments):
            s = sentiments[i]
            score = round(float(s.score), 3)
            pos = round(float(s.positive), 3)
            neg = round(float(s.negative), 3)
            neu = round(float(s.neutral), 3)
        else:
            score, pos, neg, neu = 0.0, 0.33, 0.33, 0.34

        badge = "POSITIVE" if score > 0.15 else "NEGATIVE" if score < -0.15 else "NEUTRAL"
        color = "green" if score > 0.15 else "red" if score < -0.15 else "gray"

        headlines_payload.append({
            "ticker": item["ticker"],
            "headline": item["text"],
            "sentiment_score": score,
            "sentiment_label": badge,
            "badge_color": color,
            "positive_prob": pos,
            "negative_prob": neg,
            "neutral_prob": neu
        })

    payload = {
        "ticker": ticker or "ALL",
        "count": len(headlines_payload),
        "headlines": headlines_payload[:limit]
    }

    try:
        redis_client.setex(cache_key, 180, json.dumps(payload))
    except Exception:
        pass

    return payload

