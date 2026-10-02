from celery import Celery
from celery.schedules import crontab
import sys
import time
import ssl  
from pathlib import Path

REDIS_URL = "rediss://..."  # keep your existing value (better: read it from an env var)

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
            "schedule": crontab(hour=16, minute=0, day_of_week="mon-fri"),
        },
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