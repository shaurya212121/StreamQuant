from celery import Celery
from celery.schedules import crontab
import random
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


def redis_ssl_settings(url: str) -> dict:
    """Celery TLS settings for the given Redis URL.

    TLS is enabled only for rediss:// URLs (e.g. Upstash), with certificate
    verification on. Plain redis:// (local Docker Redis) gets no TLS settings,
    because Celery refuses to start when SSL options are paired with redis://.
    """
    if not url.startswith("rediss://"):
        return {}
    tls = {"ssl_cert_reqs": ssl.CERT_REQUIRED}
    return {"broker_use_ssl": tls, "redis_backend_use_ssl": tls}


celery_app = Celery(
    "streamquant_worker",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    **redis_ssl_settings(REDIS_URL),
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


# Ingestion retry policy: 60s, 120s, 240s, 480s, 960s (plus jitter) ≈ 31 min worst case,
# long enough to ride out a Yahoo Finance rate-limit window.
INGEST_MAX_RETRIES = 5
INGEST_BACKOFF_BASE_SECONDS = 60
INGEST_BACKOFF_MAX_SECONDS = 30 * 60


def retry_countdown(retries: int) -> float:
    """Exponential backoff with jitter for the given number of retries so far.

    The delay doubles each attempt (capped), and half of it is randomised so
    tickers throttled at the same moment don't all retry in lock-step.
    """
    delay = min(INGEST_BACKOFF_BASE_SECONDS * (2 ** retries), INGEST_BACKOFF_MAX_SECONDS)
    return delay / 2 + random.uniform(0, delay / 2)


def _ensure_backend_on_path():
    # data_engine uses top-level imports (cleaner, stock), so backend/ must be on sys.path
    backend_dir = str(Path(__file__).resolve().parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)


@celery_app.task(bind=True, max_retries=INGEST_MAX_RETRIES)
def ingest_ticker_task(self, symbol: str):
    """Fetch and store missing bars for one ticker, retrying transient failures with backoff."""
    _ensure_backend_on_path()
    from data_engine import TRANSIENT_ERRORS, ingest_ticker

    try:
        return {"ticker": symbol, "result": ingest_ticker(symbol)}
    except TRANSIENT_ERRORS as exc:
        countdown = retry_countdown(self.request.retries)
        print(
            f"[WORKER] ⏳ {symbol} ingestion failed ({exc}); retry "
            f"{self.request.retries + 1}/{INGEST_MAX_RETRIES} in {countdown:.0f}s"
        )
        raise self.retry(exc=exc, countdown=countdown)


@celery_app.task
def run_daily_data_pipeline():
    """Market-close job: queue one independently-retried ingestion task per ticker.

    Splitting per ticker means a throttled or failing ticker retries on its own
    schedule without blocking or re-running the tickers that already succeeded.
    """
    _ensure_backend_on_path()
    from data_engine import TICKERS

    return {symbol: ingest_ticker_task.delay(symbol).id for symbol in TICKERS}

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