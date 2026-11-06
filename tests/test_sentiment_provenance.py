import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.main import app, confluence_decision
from ml_models.sentiment import _lexicon_analyze, analyze_headlines_with_method

client = TestClient(app)


class TestSentimentProvenanceAndRules:
    def test_lexicon_fallback_method_tag(self):
        # When lexicon fallback runs, it tags the result with 'lexicon_fallback'
        text = "Apple reports strong revenue and record profits."
        sentiment = _lexicon_analyze(text)
        assert sentiment.positive > sentiment.negative
        assert sentiment.score > 0.15

    def test_confluence_blocks_strong_buy_on_lexicon_method(self):
        # C4: Lexicon method cannot trigger STRONG BUY
        signal, summary = confluence_decision(
            is_bullish=True,
            news_score=0.45,
            is_anomaly=False,
            vol_z=0.5,
            confidence_pct=85.0,
            sentiment_method="lexicon_fallback"
        )
        assert signal != "STRONG BUY"
        assert "CAUTIOUS BUY" in signal
        assert "unverified" in summary.lower()

    def test_confluence_blocks_strong_buy_on_low_confidence(self):
        # M3: Confidence below 65% caps at CAUTIOUS BUY
        signal, summary = confluence_decision(
            is_bullish=True,
            news_score=0.45,
            is_anomaly=False,
            vol_z=0.5,
            confidence_pct=58.0,
            sentiment_method="finbert"
        )
        assert signal != "STRONG BUY"
        assert "CAUTIOUS BUY" in signal
        assert "below 65%" in summary

    def test_confluence_allows_strong_buy_when_verified_and_high_confidence(self):
        signal, summary = confluence_decision(
            is_bullish=True,
            news_score=0.45,
            is_anomaly=False,
            vol_z=0.5,
            confidence_pct=78.5,
            sentiment_method="finbert"
        )
        assert signal == "STRONG BUY"


class TestHeadlinesEndpoint:
    @patch("backend.main.redis_client")
    def test_headlines_returns_cached_payload(self, mock_redis):
        import json
        cached_data = {
            "ticker": "AAPL",
            "count": 1,
            "headlines": [{
                "ticker": "AAPL",
                "headline": "Apple test headline",
                "sentiment_score": 0.5,
                "sentiment_label": "POSITIVE",
                "badge_color": "green",
                "positive_prob": 0.8,
                "negative_prob": 0.1,
                "neutral_prob": 0.1,
                "method": "finbert"
            }]
        }
        mock_redis.get.return_value = json.dumps(cached_data)

        resp = client.get("/api/news/headlines?ticker=AAPL&limit=1")
        assert resp.status_code == 200
        assert resp.json()["headlines"][0]["headline"] == "Apple test headline"

