"""Prediction endpoints: all 8 trained features, strict UNAVAILABLE, no fabricated values.

Uses the real trained model files in ml_models/; only the database, Redis and
Yahoo are faked.
"""
import datetime
import math

import pytest
from fastapi.testclient import TestClient

from backend import main
from db.models import StockNewsSentiment, StockPrice

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
FABRICATED_VALUES = {200.0, 82.5, 76.0, 75.0, 70.0}


def make_price(date=datetime.date(2026, 10, 23), close=230.0, **overrides):
    values = dict(
        ticker="AAPL", date=date, open=close, high=close * 1.01, low=close * 0.99, close=close,
        volume=50_000_000, sma_ratio=0.02, macd=1.5, rsi=55.0, daily_return=0.004,
        volume_z_score=0.8, rolling_sharpe=1.2, rolling_max_drawdown=-0.03, atr_20=3.1,
    )
    values.update(overrides)
    return StockPrice(**values)


def make_sentiment(score=0.3):
    return StockNewsSentiment(
        ticker="AAPL", date=datetime.date(2026, 10, 23), sentiment_score=score,
        positive_prob=0.5, negative_prob=0.2, neutral_prob=0.3, headline_count=10,
    )


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *args):
        return self

    def order_by(self, *args):
        return self

    def limit(self, n):
        self.rows = self.rows[:n]
        return self

    def all(self):
        return self.rows

    def first(self):
        return self.rows[0] if self.rows else None


class FakeDB:
    """Returns the same newest-first price rows for every ticker."""

    def __init__(self, prices, sentiment=None):
        self.prices = prices
        self.sentiment = sentiment

    def query(self, entity):
        if entity is StockPrice:
            return FakeQuery(list(self.prices))
        return FakeQuery([self.sentiment] if self.sentiment else [])

    def rollback(self):
        pass


class FakeRedis:
    def __init__(self):
        self.cached_keys = []

    def get(self, key):
        return None

    def setex(self, key, ttl, value):
        self.cached_keys.append(key)

    def incr(self, key):
        return 1

    def expire(self, key, ttl):
        pass

    def ttl(self, key):
        return 60

    def ping(self):
        return True


@pytest.fixture
def api(monkeypatch):
    """Test client wired to a fake DB/Redis; returns (client, set_db, fake_redis)."""
    fake_redis = FakeRedis()
    monkeypatch.setattr(main, "redis_client", fake_redis)
    monkeypatch.setattr(main, "_yfinance_price_fields", lambda sym: None)

    state = {"db": FakeDB([])}
    main.app.dependency_overrides[main.get_db] = lambda: state["db"]

    def set_db(prices, sentiment=None):
        state["db"] = FakeDB(prices, sentiment)

    yield TestClient(main.app), set_db, fake_redis
    main.app.dependency_overrides.clear()


def full_history():
    return [make_price(), make_price(date=datetime.date(2026, 10, 22), close=228.0)]


# --- the 5-vs-8 feature bug ---------------------------------------------------

def test_every_model_receives_all_its_trained_features_in_order(api, monkeypatch):
    client, set_db, _ = api
    set_db(full_history(), make_sentiment())

    seen = {}
    real_predict_trend = main.predict_trend

    def spy(model, features):
        seen[tuple(model.get_booster().feature_names)] = list(features.columns)
        return real_predict_trend(model, features)

    monkeypatch.setattr(main, "predict_trend", spy)
    rows = client.get("/api/market/overview").json()

    assert [r["status"] for r in rows] == ["OK"] * 6
    assert seen, "no model was called"
    for trained, received in seen.items():
        assert received == list(trained)
        assert len(received) == 8


def test_overview_confidence_comes_from_the_model(api):
    client, set_db, _ = api
    set_db(full_history(), make_sentiment())

    rows = client.get("/api/market/overview").json()

    for row in rows:
        model = main.get_loaded_model(row["ticker"])
        features, _ = main.build_feature_frame(make_price(), model)
        _, expected = main.predict_trend(model, features)
        assert row["confidence_pct"] == expected


# --- strict UNAVAILABLE instead of fabricated values --------------------------

def test_overview_missing_feature_is_unavailable_not_guessed(api):
    client, set_db, cache = api
    set_db([make_price(atr_20=None), make_price(date=datetime.date(2026, 10, 22))])

    rows = client.get("/api/market/overview").json()

    for row in rows:
        assert row["status"] == "UNAVAILABLE"
        assert row["confluence_signal"] == "UNAVAILABLE"
        assert row["confidence_pct"] is None
        assert "atr_20" in row["reason"]
        assert row["latest_price"] == 230.0  # real price is still shown
    assert cache.cached_keys == []  # never cache an UNAVAILABLE grid


def test_overview_with_no_data_has_no_fabricated_price(api):
    client, set_db, _ = api
    set_db([])

    rows = client.get("/api/market/overview").json()

    assert len(rows) == 6
    for row in rows:
        assert row["status"] == "UNAVAILABLE"
        assert row["latest_price"] is None
        assert row["confidence_pct"] is None


def test_overview_uses_real_yahoo_prices_but_never_yahoo_signals(api, monkeypatch):
    client, set_db, _ = api
    set_db([])
    monkeypatch.setattr(
        main, "_yfinance_price_fields",
        lambda sym: main._price_fields(101.0, 100.0, "2026-10-23", "yfinance"),
    )

    rows = client.get("/api/market/overview").json()

    for row in rows:
        assert row["latest_price"] == 101.0
        assert row["price_source"] == "yfinance"
        assert row["confluence_signal"] == "UNAVAILABLE"
        assert row["confidence_pct"] is None


def test_overview_without_model_is_unavailable(api, monkeypatch):
    client, set_db, _ = api
    set_db(full_history(), make_sentiment())
    monkeypatch.setattr(main, "get_loaded_model", lambda ticker: None)

    rows = client.get("/api/market/overview").json()

    assert all(r["status"] == "UNAVAILABLE" and "No trained model" in r["reason"] for r in rows)


@pytest.mark.parametrize("prices", [
    lambda: full_history(),
    lambda: [make_price(rsi=None), make_price(date=datetime.date(2026, 10, 22))],
    lambda: [],
])
def test_overview_never_contains_fabricated_numbers(api, prices):
    client, set_db, _ = api
    set_db(prices(), make_sentiment())

    for row in client.get("/api/market/overview").json():
        assert row["latest_price"] not in FABRICATED_VALUES
        assert row["confidence_pct"] not in FABRICATED_VALUES or row["status"] == "OK"


def test_overview_uses_stored_zscore_and_api_threshold(api):
    """2.5σ was a 'spike' under the old 2.0 threshold; the API threshold is 3.0."""
    client, set_db, _ = api
    set_db([make_price(volume_z_score=2.5), make_price(date=datetime.date(2026, 10, 22))], make_sentiment())

    for row in client.get("/api/market/overview").json():
        assert row["volume_z_score"] == 2.5
        assert row["is_anomaly"] is False


def test_prediction_missing_feature_is_unavailable(api):
    client, set_db, cache = api
    set_db([make_price(rolling_sharpe=None)])

    data = client.get("/api/stocks/AAPL/prediction").json()

    assert data["status"] == "UNAVAILABLE"
    assert data["trend_prediction"] == "UNAVAILABLE"
    assert data["confidence_pct"] is None
    assert data["missing_features"] == ["rolling_sharpe"]
    assert cache.cached_keys == []


def test_prediction_ok_path(api):
    client, set_db, cache = api
    set_db([make_price()])

    data = client.get("/api/stocks/AAPL/prediction").json()

    assert data["status"] == "OK"
    assert data["trend_prediction"] in {"BULLISH", "BEARISH"}
    assert 50.0 <= data["confidence_pct"] <= 100.0
    assert cache.cached_keys == ["prediction:AAPL"]


def test_intelligence_missing_feature_keeps_real_layers(api):
    client, set_db, cache = api
    set_db([make_price(macd=None)], make_sentiment())

    data = client.get("/api/stocks/AAPL/intelligence").json()

    assert data["status"] == "UNAVAILABLE"
    assert data["confluence_decision"]["signal"] == "UNAVAILABLE"
    assert data["ml_trend_layer"]["confidence_pct"] is None
    assert data["ml_trend_layer"]["top_drivers"] == []
    assert data["statistical_anomaly_layer"]["volume_z_score"] == 0.8  # real data still returned
    assert data["nlp_sentiment_layer"]["status"] == "AVAILABLE"
    assert cache.cached_keys == []


def test_intelligence_ok_path(api):
    client, set_db, _ = api
    set_db([make_price()], make_sentiment())

    data = client.get("/api/stocks/AAPL/intelligence").json()

    assert data["status"] == "OK"
    assert data["ml_trend_layer"]["prediction"] in {"BULLISH", "BEARISH"}
    assert data["confluence_decision"]["signal"] != "UNAVAILABLE"


# --- feature builder ----------------------------------------------------------

def test_real_zero_rsi_is_not_replaced_with_50():
    """The old `rsi or 50.0` turned a genuine RSI of 0 into 50."""
    model = main.get_loaded_model("AAPL")
    features, missing = main.build_feature_frame(make_price(rsi=0.0), model)
    assert missing == []
    assert features["rsi"].iloc[0] == 0.0


def test_nan_feature_counts_as_missing():
    model = main.get_loaded_model("AAPL")
    features, missing = main.build_feature_frame(make_price(macd=math.nan), model)
    assert features is None
    assert missing == ["macd"]
