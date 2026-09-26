from celery import Celery
import time
import ssl  # <--- We added the core Python SSL library

REDIS_URL = "rediss://default:gQAAAAAABH-sAAIgcDJlNDZmNTE2OTgyYjE0OThkODMyODlmNDZjZjk1Njk1MA@summary-fox-294828.upstash.io:6379"

celery_app = Celery(
    "streamquant_worker",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Force Celery to accept the Upstash Cloud certificate
celery_app.conf.update(
    broker_use_ssl={'ssl_cert_reqs': ssl.CERT_NONE},
    redis_backend_use_ssl={'ssl_cert_reqs': ssl.CERT_NONE}
)

@celery_app.task
def run_heavy_ml_model(ticker: str):
    print(f"\n[WORKER] 🚀 Starting heavy ML Pipeline for {ticker}...")
    time.sleep(10)
    print(f"[WORKER] ✅ Finished ML Pipeline for {ticker}!")
    return {"ticker": ticker, "status": "ML_COMPLETE"}