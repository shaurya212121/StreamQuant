import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import os

# Set testing API key
os.environ["STREAMQUANT_API_KEY"] = "test-secret-key-12345"

from backend.main import app
from backend.execution import AlpacaExecutionEngine

client = TestClient(app)
AUTH_HEADERS = {"X-API-KEY": "test-secret-key-12345"}


class TestTradeAuthentication:
    def test_missing_api_key_rejected(self):
        resp = client.post("/execute-trade", json={"symbol": "AAPL"})
        assert resp.status_code == 401
        assert "Unauthorized" in resp.json()["detail"]

    def test_invalid_api_key_rejected(self):
        resp = client.post("/execute-trade", json={"symbol": "AAPL"}, headers={"X-API-KEY": "wrong-key"})
        assert resp.status_code == 401
        assert "Unauthorized" in resp.json()["detail"]

    def test_portfolio_summary_requires_auth(self):
        resp = client.get("/portfolio-summary")
        assert resp.status_code == 401

        resp_auth = client.get("/portfolio-summary", headers=AUTH_HEADERS)
        assert resp_auth.status_code == 200

    def test_recent_orders_requires_auth(self):
        resp = client.get("/recent-orders")
        assert resp.status_code == 401

        resp_auth = client.get("/recent-orders", headers=AUTH_HEADERS)
        assert resp_auth.status_code == 200


class TestTradeValidationAndRiskFirewall:
    def test_unsupported_symbol_rejected(self):
        resp = client.post(
            "/execute-trade",
            json={"symbol": "INVALID", "signal": "STRONG BUY", "confidence_pct": 99.0},
            headers=AUTH_HEADERS
        )
        assert resp.status_code == 422 or resp.status_code == 400

    @patch("backend.main.compute_unified_intelligence_data")
    def test_weak_signal_rejected_by_server(self, mock_intel):
        # Server verifies signal is BEARISH / CAUTIOUS BUY, not STRONG BUY
        mock_intel.return_value = {
            "status": "OK",
            "confluence_decision": {"signal": "CAUTIOUS BUY", "summary": "Weak conviction"},
            "ml_trend_layer": {"confidence_pct": 55.0},
            "nlp_sentiment_layer": {"sentiment_score": 0.10, "method": "finbert"}
        }

        # Even if client tries to send "STRONG BUY", server computes and rejects it
        resp = client.post(
            "/execute-trade",
            json={"symbol": "AAPL", "signal": "STRONG BUY", "confidence_pct": 99.0},
            headers=AUTH_HEADERS
        )
        assert resp.status_code == 400
        assert "Execution strictly requires STRONG BUY" in resp.json()["detail"]

    @patch("backend.main.compute_unified_intelligence_data")
    def test_lexicon_fallback_cannot_execute_strong_buy(self, mock_intel):
        # Lexicon fallback sentiment cannot execute STRONG BUY
        mock_intel.return_value = {
            "status": "OK",
            "confluence_decision": {"signal": "CAUTIOUS BUY", "summary": "Unverified sentiment"},
            "ml_trend_layer": {"confidence_pct": 85.0},
            "nlp_sentiment_layer": {"sentiment_score": 0.60, "method": "lexicon_fallback"}
        }

        resp = client.post(
            "/execute-trade",
            json={"symbol": "AAPL"},
            headers=AUTH_HEADERS
        )
        assert resp.status_code == 400


class TestExecutionEngineGuards:
    @pytest.fixture
    def mock_engine(self):
        engine = AlpacaExecutionEngine()
        engine.client = MagicMock()
        engine.data_client = MagicMock()
        return engine

    def test_market_hours_fail_closed_on_error(self, mock_engine):
        # When clock lookup errors out, execution must fail closed
        mock_engine.get_market_clock = MagicMock(return_value={"status": "error", "is_open": False})
        res = mock_engine.execute_trade_signal(symbol="AAPL", signal="STRONG BUY")
        assert res["success"] is False
        assert "Exchange clock lookup failed" in res["error"]

    def test_market_hours_blocks_when_closed(self, mock_engine):
        mock_engine.get_market_clock = MagicMock(return_value={"status": "online", "is_open": False, "next_open": "09:30"})
        res = mock_engine.execute_trade_signal(symbol="AAPL", signal="STRONG BUY")
        assert res["success"] is False
        assert "Market is currently closed" in res["error"]

    def test_position_aware_sizing_enforces_cap(self, mock_engine):
        # Portfolio: $100k equity. 5% cap = $5,000 max.
        mock_engine.get_account_portfolio = MagicMock(return_value={
            "status": "ACTIVE",
            "portfolio_value": 100000.0,
            "buying_power": 200000.0,
            "cash": 50000.0
        })
        # Already holding $4,000 worth of AAPL
        mock_engine.get_open_position = MagicMock(return_value={
            "symbol": "AAPL",
            "qty": 20.0,
            "market_value": 4000.0
        })

        # At $200/share, room is only $1,000 (5 shares max)
        sizing = mock_engine.calculate_position_size(
            symbol="AAPL",
            current_price=200.0,
            confidence_pct=100.0
        )
        assert sizing["qty"] <= 5
        assert sizing["target_dollars"] <= 1000.0

    def test_position_cap_rejects_when_already_full(self, mock_engine):
        # Portfolio: $100k equity. Holding $5,500 of AAPL (at or over 5% cap)
        mock_engine.get_account_portfolio = MagicMock(return_value={
            "status": "ACTIVE",
            "portfolio_value": 100000.0,
            "buying_power": 200000.0
        })
        mock_engine.get_open_position = MagicMock(return_value={
            "symbol": "AAPL",
            "qty": 27.5,
            "market_value": 5500.0
        })

        sizing = mock_engine.calculate_position_size(
            symbol="AAPL",
            current_price=200.0,
            confidence_pct=100.0
        )
        assert sizing["qty"] == 0
        assert "Position limit reached" in sizing["reason"]

    def test_idempotency_key_generated(self, mock_engine):
        mock_engine.get_market_clock = MagicMock(return_value={"status": "online", "is_open": True})
        mock_engine.get_account_portfolio = MagicMock(return_value={
            "status": "ACTIVE",
            "portfolio_value": 100000.0,
            "buying_power": 200000.0
        })
        mock_engine.get_latest_market_price = MagicMock(return_value=150.0)
        mock_engine.get_open_position = MagicMock(return_value=None)

        mock_order = MagicMock()
        mock_order.id = "order-12345"
        mock_order.symbol = "AAPL"
        mock_order.status = "filled"
        mock_order.qty = 10
        mock_engine.client.submit_order = MagicMock(return_value=mock_order)

        res = mock_engine.execute_trade_signal(symbol="AAPL", signal="STRONG BUY", qty=10, sizing_mode="manual")
        assert res["success"] is True
        assert res["client_order_id"].startswith("sq_aapl_")
        assert res["executed_price"] == 150.0

    def test_kill_switch_blocks_orders(self, mock_engine, monkeypatch):
        monkeypatch.setenv("STREAMQUANT_KILL_SWITCH", "true")
        res = mock_engine.execute_trade_signal(symbol="AAPL", signal="STRONG BUY")
        assert res["success"] is False
        assert "Kill Switch" in res["error"]
        assert res["status_code"] == 503

    def test_daily_loss_circuit_breaker(self, mock_engine):
        mock_engine.get_market_clock = MagicMock(return_value={"status": "online", "is_open": True})
        # Last equity $100k, current equity $95k (-5% daily loss)
        mock_engine.get_account_portfolio = MagicMock(return_value={
            "status": "ACTIVE",
            "portfolio_value": 95000.0,
            "last_equity": 100000.0,
            "buying_power": 190000.0
        })
        res = mock_engine.execute_trade_signal(symbol="AAPL", signal="STRONG BUY", qty=5, sizing_mode="manual")
        assert res["success"] is False
        assert "Daily loss limit (-3%) triggered" in res["error"]

    def test_manual_fractional_shares_rejected(self, mock_engine):
        mock_engine.get_market_clock = MagicMock(return_value={"status": "online", "is_open": True})
        mock_engine.get_account_portfolio = MagicMock(return_value={
            "status": "ACTIVE",
            "portfolio_value": 100000.0,
            "last_equity": 100000.0,
            "buying_power": 200000.0
        })
        res = mock_engine.execute_trade_signal(symbol="AAPL", signal="STRONG BUY", qty=5.5, sizing_mode="manual")
        assert res["success"] is False
        assert "positive whole integer" in res["error"]

    def test_stale_sentiment_downgrades_strong_buy(self):
        from backend.main import confluence_decision
        # Sentiment older than 3 days
        signal, summary = confluence_decision(
            is_bullish=True,
            news_score=0.45,
            is_anomaly=False,
            vol_z=0.2,
            confidence_pct=85.0,
            sentiment_method="finbert",
            is_sentiment_stale=True
        )
        assert signal == "CAUTIOUS BUY (STALE SENTIMENT)"

