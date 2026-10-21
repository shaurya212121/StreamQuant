"""
StreamQuant Distributed High-Concurrency Load Testing Suite
===========================================================
Simulates concurrent virtual users hitting the FastAPI gateway,
evaluating Redis caching latency and distributed rate limiting defenses.
"""

import random
from locust import HttpUser, task, between

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]


class StreamQuantUser(HttpUser):
    host = "http://127.0.0.1:8000"
    wait_time = between(0.3, 1.0)

    @task(7)
    def test_unified_intelligence_gateway(self):
        """
        Hits the Redis-cached Unified Intelligence Gateway.
        Catches HTTP 429 as expected rate-limiting security behavior.
        """
        ticker = random.choice(TICKERS)
        with self.client.get(
            f"/api/stocks/{ticker}/intelligence",
            name="/api/stocks/[ticker]/intelligence",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                # 429 confirms our Redis sliding-window defense is working!
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(3)
    def test_stock_price_history(self):
        """
        Hits PostgreSQL historical queries to test database connection pooling.
        """
        ticker = random.choice(TICKERS)
        with self.client.get(
            f"/api/stocks/{ticker}?limit=30",
            name="/api/stocks/[ticker]?limit=30",
            catch_response=True
        ) as response:
            if response.status_code in [200, 429]:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
