import ssl
import sys
from pathlib import Path

import pytest

# data_engine uses top-level imports (cleaner, stock), so backend/ must be on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import data_engine  # noqa: E402
from backend import worker  # noqa: E402


# --- Redis TLS settings (C3) --------------------------------------------------

def test_plain_redis_url_gets_no_tls_settings():
    """Local Docker Redis (redis://) must not get SSL options, or Celery refuses to start."""
    assert worker.redis_ssl_settings("redis://redis:6379/0") == {}


def test_tls_redis_url_verifies_certificates():
    settings = worker.redis_ssl_settings("rediss://default:pw@host.upstash.io:6379")
    assert settings["broker_use_ssl"]["ssl_cert_reqs"] == ssl.CERT_REQUIRED
    assert settings["redis_backend_use_ssl"]["ssl_cert_reqs"] == ssl.CERT_REQUIRED


# --- backoff schedule ---------------------------------------------------------

@pytest.mark.parametrize("retries, full_delay", [(0, 60), (1, 120), (2, 240), (3, 480), (4, 960)])
def test_backoff_doubles_each_retry(retries, full_delay):
    for _ in range(50):
        countdown = worker.retry_countdown(retries)
        assert full_delay / 2 <= countdown <= full_delay


def test_backoff_is_capped():
    for _ in range(50):
        assert worker.retry_countdown(20) <= worker.INGEST_BACKOFF_MAX_SECONDS


def test_backoff_has_jitter():
    """Tickers throttled together should not all retry at the same instant."""
    assert len({round(worker.retry_countdown(2), 6) for _ in range(20)}) > 1


# --- retry behaviour (eager mode: retries run immediately, no sleeping) -------

def _fail_then_succeed(monkeypatch, failures, exc_type):
    calls = []

    def fake_ingest(symbol, overlap_bars=data_engine.OVERLAP_BARS):
        calls.append(symbol)
        if len(calls) <= failures:
            raise exc_type("simulated failure")
        return "ok: 1 new bar(s)"

    monkeypatch.setattr(data_engine, "ingest_ticker", fake_ingest)
    return calls


def test_transient_failure_is_retried_until_success(monkeypatch):
    """Yahoo returns nothing twice (rate limit), then recovers: the task succeeds."""
    calls = _fail_then_succeed(monkeypatch, failures=2, exc_type=data_engine.DataFetchError)

    result = worker.ingest_ticker_task.apply(args=["AAPL"])

    assert result.successful()
    assert result.result == {"ticker": "AAPL", "result": "ok: 1 new bar(s)"}
    assert len(calls) == 3


def test_retries_stop_after_max_retries(monkeypatch):
    calls = _fail_then_succeed(monkeypatch, failures=99, exc_type=data_engine.DataFetchError)

    result = worker.ingest_ticker_task.apply(args=["AAPL"])

    assert result.failed()
    assert isinstance(result.result, data_engine.DataFetchError)
    assert len(calls) == 1 + worker.INGEST_MAX_RETRIES


def test_non_transient_failure_is_not_retried(monkeypatch):
    """A bug (e.g. ValueError) fails fast instead of burning 30 minutes of retries."""
    calls = _fail_then_succeed(monkeypatch, failures=99, exc_type=ValueError)

    result = worker.ingest_ticker_task.apply(args=["AAPL"])

    assert result.failed()
    assert len(calls) == 1


def test_daily_job_queues_one_task_per_ticker(monkeypatch):
    queued = []

    class FakeAsyncResult:
        def __init__(self, symbol):
            self.id = f"task-{symbol}"

    def fake_delay(symbol):
        queued.append(symbol)
        return FakeAsyncResult(symbol)

    monkeypatch.setattr(worker.ingest_ticker_task, "delay", fake_delay)

    result = worker.run_daily_data_pipeline.apply().result

    assert queued == data_engine.TICKERS
    assert result == {s: f"task-{s}" for s in data_engine.TICKERS}
