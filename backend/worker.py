from celery import Celery
from celery.schedules import crontab
import sys
import time
import ssl  
import os
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load root .env file
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

celery_app = Celery(
    "streamquant_worker",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    broker_use_ssl={'ssl_cert_reqs': ssl.CERT_NONE},
    redis_backend_use_ssl={'ssl_cert_reqs': ssl.CERT_NONE},
    # Daily market-close ingestion (US equities close 4:00 PM New York time, DST-aware)
    timezone="America/New_York",
    enable_utc=True,
    beat_schedule={
        "daily-market-close-ingestion": {
            "task": "backend.worker.run_daily_data_pipeline",
            # FLAW #9 FIX: Shifted minute=0 to minute=15 to bypass the 4:00 PM closing auction data race
            "schedule": crontab(hour=16, minute=15, day_of_week="mon-fri"),
        },
        "daily-sentiment-batch-processing": {
            "task": "backend.worker.run_all_symbols_sentiment",
            # Run nightly to pre-compute FinBERT sentiment for the next day
            "schedule": crontab(hour=2, minute=0),
        }
    },
)

@celery_app.task
def run_heavy_ml_model(ticker: str):
    print(f"\n[WORKER] 🚀 Starting heavy ML Pipeline for {ticker}...")
    time.sleep(10)
    print(f"[WORKER] ✅ Finished ML Pipeline for {ticker}!")
    return {"ticker": ticker, "status": "ML_COMPLETE"}


@celery_app.task(bind=True, max_retries=2, default_retry_delay=300)
def run_daily_data_pipeline(self):
    """Fetch latest day's data, validate with Pydantic, upsert into Supabase."""
    # data_engine uses top-level imports (cleaner, stock), so backend/ must be on sys.path
    backend_dir = str(Path(__file__).resolve().parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from data_engine import run_daily_pipeline
    try:
        return run_daily_pipeline()
    except Exception as exc:
        raise self.retry(exc=exc)

# FLAW #11 FIX: Decoupled FinBERT execution background task
@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def run_batch_sentiment(self, ticker: str):
    """
    Executes the heavy FinBERT sentiment model entirely in the background,
    saving the result to PostgreSQL to avoid blocking the FastAPI event loop.
    """
    ml_dir = str(Path(__file__).resolve().parent.parent / "ml_models")
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
    from ml_models.sentiment import analyze_and_store
    
    print(f"\n[WORKER] 🚀 Starting Background Sentiment Analysis for {ticker}...")
    try:
        # Executes ML inference and directly writes to DB
        result = analyze_and_store(ticker)
        print(f"[WORKER] ✅ Finished Sentiment Analysis for {ticker}!")
        return {"ticker": ticker, "status": "SENTIMENT_COMPLETE"}
    except Exception as exc:
        print(f"[WORKER] ❌ Failed Sentiment Analysis for {ticker}: {exc}")
        raise self.retry(exc=exc)

@celery_app.task
def run_all_symbols_sentiment():
    active_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
    for sym in active_symbols:
        run_batch_sentiment.delay(sym)